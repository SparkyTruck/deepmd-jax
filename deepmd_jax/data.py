import numpy as np
import jax.numpy as jnp
from jax import vmap
from glob import glob
from os.path import abspath
from ase.io import read
from .utils import shift, get_relative_coord, sr


def _classify_path(p):
    return 'extxyz' if isinstance(p, str) and p.lower().endswith(('.xyz', '.extxyz')) else 'dp'


def _flatten_paths(paths):
    for p in paths:
        if isinstance(p, list):
            yield from _flatten_paths(p)
        else:
            yield p


def Dataset(paths, labels, params=None, chemical_types=None):
    """
    Create a dataset object from file paths.

    This function dispatches to the appropriate dataset class based on the input paths
    and formats. It supports DeepMD (DP) format (.npy files in directories) and
    extended XYZ format (.xyz/.extxyz files).

    Parameters
    ----------
    paths : str or list of str
        File paths to load data from. For DP format, paths should be strings pointing
        to directories containing 'type.raw' and 'set.*/' subdirectories with .npy files.
        For extxyz format, paths should be strings ending in '.xyz' or '.extxyz'.
        If paths is a list, it creates a composite dataset from multiple subsets.
    labels : list of str
        List of data labels to load, e.g., ['coord', 'energy', 'force'].
    params : dict, optional
        Additional parameters for dataset configuration, such as 'atomic_sel'.
    chemical_types : list of int, optional
        List of atomic numbers defining the chemical types. If None, inferred from data.

    Returns
    -------
    DatasetLeaf, DatasetGroup, ExtXYZDataset, or DPDataset
        - DatasetLeaf: For single composition groups.
        - DatasetGroup: For composite datasets with multiple subsets.
        - ExtXYZDataset: For extxyz files, groups frames by composition into DatasetLeaf subsets.
        - DPDataset: For a DP format directory.

    Constraints:
    - Mixing DP directories and extxyz files in the same paths list is not supported.
    """
    flat_paths = [paths] if isinstance(paths, str) else list(_flatten_paths(paths))
    if len(flat_paths) == 1:
        path = flat_paths[0]
        if _classify_path(path) == 'extxyz':
            return ExtXYZDataset([path], labels, params, chemical_types)
        return DPDataset(path, labels, params, chemical_types)

    formats = {_classify_path(p) for p in flat_paths}
    if len(formats) > 1:
        raise ValueError('Mixing DP and extxyz paths is not supported: %s' % (paths,))

    if formats == {'extxyz'}:
        return ExtXYZDataset(flat_paths, labels, params, chemical_types)
    leaves = [DPDataset(p, labels, params, chemical_types) for p in flat_paths]
    return DatasetGroup(leaves, chemical_types)


class DatasetLeaf:
    """
    Internal dataset class for in-memory data of a single composition.

    Data is stored in the input atom order. The model receives ``type_idx`` and
    handles type sorting internally.
    """
    def __init__(self, labels, params, type_arr, data, paths=None):
        self.chemical_types = getattr(self, 'chemical_types', None)
        self.type_idx = np.array(type_arr, dtype=int)
        self.type = self.type_idx
        self.data = data
        self.natoms = len(self.type_idx)
        self.nframes = len(self.data['coord'])
        for l in labels:
            assert self.data[l].shape[0] == self.nframes, \
                f"{l}.npy has {self.data[l].shape[0]} frames, expected {self.nframes}"
        self.pointer = self.nframes
        self.type_count = np.array([(self.type_idx == i).sum() for i in range(max(self.type_idx) + 1)])
        self.ntypes = len(self.type_count)
        self.valid_types = np.arange(self.ntypes)
        self.nsel = params.get('atomic_sel', None)
        if self.nsel is not None:
            self.nsel = [i for i in self.nsel if i in range(self.ntypes)]
        if any(['atomic' in l for l in labels]):
            self.nlabels = sum(self.type_count[self.nsel])
        else:
            self.nlabels = self.natoms
        for l in labels:
            if l in ['coord', 'force']:
                self.data[l] = self.data[l].reshape(self.data[l].shape[0], -1, 3)
            if l == 'energy':
                self.data[l] = self.data[l].reshape(-1)
            if 'atomic' in l:
                try:
                    self.data[l] = self.data[l].reshape(self.data[l].shape[0], self.nlabels, -1)
                    assert self.data[l].shape[2] in (3, 9)
                except Exception:
                    raise ValueError('Atomic label must have 3 (vector) or 9 (3x3 tensor) components per atom.')
        self.data['box'] = self.data['box'].reshape(-1, 3, 3)
        self.data['coord'] = np.array(vmap(shift)(self.data['coord'], self.data['box']))
        if paths is not None:
            print('# Dataset loaded: %d frames/%d atoms. Path:' % (self.nframes, self.natoms),
                  ''.join(['\n# \t\'%s\'' % abspath(path) for path in paths]))

    def count_max(self):
        return np.array(self.type_count)

    def fill_type(self, ntypes):
        self.type_count = np.pad(self.type_count, (0, ntypes - self.ntypes))

    def _get_stats(self, rcut, bs):
        if not hasattr(self, 'lattice_args'):
            raise AttributeError("lattice_args not set. Call compute_lattice_candidate(rcut) before get_stats.")
        batch = self.get_batch(bs)[0]
        coord, box = batch['coord'], batch['box']
        coord = coord[:, np.argsort(self.type_idx, kind='stable')]
        r_Bnm = vmap(get_relative_coord, (0, 0, None, None))(coord, box, self.type_count, self.lattice_args)[1]
        sr_BnM = [sr(jnp.concatenate(r, axis=-1), rcut) for r in r_Bnm]
        sr_sum = np.array([sr.sum() for sr in sr_BnM])
        sr_sum2 = np.array([(sr**2).sum() for sr in sr_BnM])
        sr_count = np.array([(sr > 1e-15).sum() for sr in sr_BnM])
        Nnbrs = (np.concatenate(sr_BnM, axis=1) > 0).sum(2).mean() + 1
        return np.array([sr_sum, sr_sum2, sr_count, Nnbrs * np.ones_like(sr_sum)])

    def get_stats(self, rcut, bs):
        self.params = {'ntypes': self.ntypes, 'rcut': rcut}
        sr_sum, sr_sum2, sr_count, Nnbrs = self._get_stats(rcut, bs)
        sr_sum, sr_sum2, sr_count = sr_sum[self.valid_types], sr_sum2[self.valid_types], sr_count[self.valid_types]
        self.params['valid_types'] = self.valid_types
        self.params['sr_mean'] = sr_sum / sr_count
        self.params['sr_std'] = np.sqrt(sr_sum2 / sr_count - self.params['sr_mean']**2)
        self.params['Nnbrs'] = Nnbrs[0]
        if self.chemical_types is not None:
            self.params['chemical_types'] = self.chemical_types
        return self.params

    def get_batch(self, batch_size, type='frame'):
        if type == 'label':
            batch_size = int(batch_size / self.nlabels + 1)
        if self.pointer + batch_size > self.nframes:
            self.pointer = 0
            perm = np.random.permutation(self.nframes)
            self.data = {l: self.data[l][perm] for l in self.data}
        batch = {
            'atomic' if 'atomic' in l else l:
            self.data[l][self.pointer:min(self.pointer + batch_size, self.nframes)]
            for l in self.data
        }
        self.pointer += batch_size
        return batch, tuple(self.type_idx), self.lattice_args

    def compute_lattice_candidate(self, rcut):
        self.lattice_args = compute_lattice_candidate(self.data['box'], rcut)

    def fit_energy(self):
        energy_stats = self._get_energy_stats()
        type_count, energy_mean = [np.array(x) for x in zip(*energy_stats)]
        type_count = type_count[:, self.valid_types]
        return np.linalg.lstsq(type_count, energy_mean, rcond=1e-3)[0].astype(np.float32)

    def get_atomic_label_scale(self):
        label = [label for label in self.data.keys() if 'atomic' in label][0]
        return np.std(self.data[label])

    def _get_energy_stats(self):
        return [(self.type_count, self.data['energy'].mean())]

    def get_flattened_data(self):
        return [{'data': self.data, 'type_idx': self.type_idx, 'lattice_args': self.lattice_args}]

    def get_leaves(self):
        return [self]


class DPDataset(DatasetLeaf):
    """
    Dataset for DeepMD (DP) format directories.

    Loads data from DP training directories containing type.raw and set.*/ subdirs
    with .npy files. Concatenates data across all sets in the directory.
    """
    def __init__(self, path, labels, params=None, chemical_types=None):
        self.chemical_types = tuple(chemical_types) if chemical_types else None
        type_arr = np.genfromtxt(path + '/type.raw').astype(int)
        data = {
            l: np.concatenate([np.load(s + l + '.npy') for s in sorted(glob(path + '/set.*/'))])
            for l in labels
        }
        super().__init__(labels, params or {}, type_arr, data, paths=[path])


class DatasetGroup:
    """
    Composite dataset made from multiple subset datasets.

    A DatasetGroup represents a mixture of DatasetLeaf subsets. Sampling across
    subsets is weighted by subset size, stored in self.prob, so larger subsets are
    selected more often during batch generation.
    """
    def __init__(self, subsets, chemical_types=None):
        self.subsets = subsets
        self.chemical_types = tuple(chemical_types) if chemical_types else None
        self.nframes = sum([subset.nframes for subset in self.subsets])
        self.ntypes = max([subset.ntypes for subset in self.subsets])
        [subset.fill_type(self.ntypes) for subset in self.subsets]
        self.prob = np.array([subset.nframes for subset in self.subsets]) / self.nframes
        self.type_count = self.count_max()
        self.valid_types = np.arange(self.ntypes)[self.type_count > 0]
        if self.chemical_types is None:
            cts = {s.chemical_types for s in self.subsets if s.chemical_types is not None}
            if len(cts) > 1:
                raise ValueError('Inconsistent chemical_types across subsets: %s' % cts)
            if cts:
                self.chemical_types = cts.pop()

    def count_max(self):
        return np.array([subset.count_max() for subset in self.subsets]).max(0)

    def fill_type(self, ntypes):
        for subset in self.subsets:
            subset.fill_type(ntypes)

    def _get_stats(self, rcut, bs):
        s = np.stack([subset._get_stats(rcut, bs) for subset in self.subsets], axis=-1)
        return (s * self.prob).sum(-1)

    def get_stats(self, rcut, bs):
        self.params = {'ntypes': self.ntypes, 'rcut': rcut}
        sr_sum, sr_sum2, sr_count, Nnbrs = self._get_stats(rcut, bs)
        sr_sum, sr_sum2, sr_count = sr_sum[self.valid_types], sr_sum2[self.valid_types], sr_count[self.valid_types]
        self.params['valid_types'] = self.valid_types
        self.params['sr_mean'] = sr_sum / sr_count
        self.params['sr_std'] = np.sqrt(sr_sum2 / sr_count - self.params['sr_mean']**2)
        self.params['Nnbrs'] = Nnbrs[0]
        if self.chemical_types is not None:
            self.params['chemical_types'] = self.chemical_types
        return self.params

    def get_batch(self, batch_size, type='frame'):
        subset = np.random.choice(len(self.subsets), p=self.prob)
        return self.subsets[subset].get_batch(batch_size, type)

    def compute_lattice_candidate(self, rcut):
        for subset in self.subsets:
            subset.compute_lattice_candidate(rcut)

    def fit_energy(self):
        energy_stats = self._get_energy_stats()
        type_count, energy_mean = [np.array(x) for x in zip(*energy_stats)]
        type_count = type_count[:, self.valid_types]
        return np.linalg.lstsq(type_count, energy_mean, rcond=1e-3)[0].astype(np.float32)

    def get_atomic_label_scale(self):
        return (np.array([subset.get_atomic_label_scale() for subset in self.subsets]) * np.array(self.prob)).sum()

    def _get_energy_stats(self):
        return sum([subset._get_energy_stats() for subset in self.subsets], [])

    def get_flattened_data(self):
        return sum([subset.get_flattened_data() for subset in self.subsets], [])

    def get_leaves(self):
        return sum([s.get_leaves() for s in self.subsets], [])


class ExtXYZDataset(DatasetGroup):
    """
    Dataset for extended XYZ (.xyz/.extxyz) files.

    Parses frames using ASE and groups them by composition before constructing
    internal DatasetLeaf subsets. Each composition group becomes one leaf.
    """
    def __init__(self, paths, labels, params=None, chemical_types=None):
        raw_frames = []
        all_zs = set()
        for path in paths:
            atoms_list = read(path, index=':')
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]
            for atoms in atoms_list:
                zs = np.asarray(atoms.get_atomic_numbers(), dtype=int)
                all_zs.update(zs.tolist())
                entry = {'_zs': zs}
                for l in labels:
                    if l == 'box':
                        entry['box'] = np.asarray(atoms.get_cell().array, dtype=np.float32)
                    elif l == 'coord':
                        entry['coord'] = np.asarray(atoms.get_positions(), dtype=np.float32)
                    elif l == 'force':
                        entry['force'] = np.asarray(atoms.get_forces(), dtype=np.float32)
                    elif l == 'energy':
                        entry['energy'] = np.asarray(atoms.get_potential_energy(), dtype=np.float32)
                    else:
                        if l in atoms.arrays:
                            entry[l] = np.asarray(atoms.arrays[l], dtype=np.float32)
                        elif l in atoms.info:
                            entry[l] = np.asarray(atoms.info[l], dtype=np.float32)
                        else:
                            raise ValueError('Label %s not found in extxyz frame from %s' % (l, path))
                raw_frames.append(entry)

        if chemical_types is None:
            chemical_types = tuple(sorted(all_zs))
        else:
            unknown = all_zs - set(chemical_types)
            if unknown:
                raise ValueError(
                    'Atomic numbers %s in extxyz files not in chemical_types=%s'
                    % (sorted(unknown), chemical_types))
        self.chemical_types = chemical_types
        z_to_idx = {z: i for i, z in enumerate(chemical_types)}
        ntypes = len(chemical_types)

        groups = {}
        for entry in raw_frames:
            zs = entry.pop('_zs')
            types = np.array([z_to_idx[int(z)] for z in zs], dtype=int)
            tc = tuple(int((np.sort(types) == i).sum()) for i in range(ntypes))
            grp = groups.setdefault(tc, {'type': types, 'frames': []})
            grp['frames'].append(entry)

        subsets = []
        for grp in groups.values():
            frames = grp['frames']
            data = {l: np.stack([f[l] for f in frames]) for l in labels}
            subsets.append(DatasetLeaf(labels, params or {}, grp['type'], data))

        super().__init__(subsets, chemical_types=chemical_types)
        print('# Dataset loaded (extxyz): %d frames in %d composition group(s). Path:'
              % (len(raw_frames), len(subsets)),
              ''.join(['\n# \t\'%s\'' % abspath(p) for p in paths]))


def compute_lattice_candidate(boxes, rcut, print_info=True, disable_ortho=False):
    N = 2  # This algorithm is heuristic and subject to change. Increase N in case of missing neighbors.
    ortho = not vmap(lambda box: box - jnp.diag(jnp.diag(box)))(boxes).any()
    recp_norm = jnp.linalg.norm((jnp.linalg.inv(boxes)), axis=-1)
    n = np.ceil(rcut * recp_norm - 0.5).astype(int).max(0)
    lattice_cand = jnp.stack(
        np.meshgrid(range(-n[0], n[0] + 1), range(-n[1], n[1] + 1), range(-n[2], n[2] + 1), indexing='ij'),
        axis=-1).reshape(-1, 3)
    trial_points = jnp.stack(np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1), np.arange(-N, N + 1)),
                             axis=-1).reshape(-1, 3) / (2 * N)
    is_neighbor = jnp.linalg.norm((lattice_cand[:, None] - trial_points)[None] @ boxes[:, None], axis=-1) < rcut
    lattice_cand = np.array(lattice_cand[is_neighbor.any((0, 2))])
    lattice_max = is_neighbor.sum(1).max().item()
    if print_info:
        print('# Lattice vectors for neighbor images: Max %d out of %d candidates.' % (lattice_max, len(lattice_cand)))
    return {'lattice_cand': tuple(map(tuple, lattice_cand)),
            'lattice_max': lattice_max,
            'ortho': ortho if not disable_ortho else False}
