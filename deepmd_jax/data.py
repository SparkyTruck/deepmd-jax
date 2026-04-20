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

class Dataset():
    def __init__(self, paths, labels, params={}, chemical_types=None, _in_memory=None):
        '''
            A dataset container supporting DP directory format and extxyz files.

            paths:
                - list of DP data directories (leaf) or list-of-lists (non-leaf); or
                - list of extxyz file paths (dispatches to from_extxyz); or
                - unused when _in_memory is provided.
            labels: data fields to load (e.g., ['coord','box','force','energy']).
            params: may contain 'atomic_sel' for atomic-label models.
            chemical_types: optional tuple of atomic numbers (Z) mapping type index -> Z.
                            Required/produced for extxyz; optional for DP format.
            _in_memory: internal — dict {'type': np.ndarray, 'data': dict} for leaves
                        constructed from in-memory data (used by from_extxyz).
        '''
        self.chemical_types = tuple(chemical_types) if chemical_types else None

        if _in_memory is not None:
            self.is_leaf = True
            self._finalize_leaf(_in_memory['type'], _in_memory['data'], labels, params, paths=None)
            return

        formats = {_classify_path(p) for p in _flatten_paths(paths)}
        if len(formats) > 1:
            raise ValueError('Mixing DP-directory and extxyz dataset paths is not supported; got both in %s' % (paths,))

        if type(paths[0]) == list:
            self.is_leaf = False
            self.subsets = [Dataset(path, labels, params, chemical_types=chemical_types)
                            for path in paths]
            self._finalize_non_leaf()
            return

        if formats == {'extxyz'}:
            self._init_from_extxyz(paths, labels, params)
            return

        self.is_leaf = True
        type_arr = np.genfromtxt(paths[0] + '/type.raw').astype(int)
        data = {l: np.concatenate(sum([[np.load(set+l+'.npy') for set in sorted(glob(path+'/set.*/'))]
                                       for path in paths], [])) for l in labels}
        self._finalize_leaf(type_arr, data, labels, params, paths=paths)

    def _finalize_leaf(self, type_arr, data, labels, params, paths=None):
        self.type = np.array(type_arr, dtype=int)
        self.data = data
        self.natoms = len(self.type)
        self.nframes = len(self.data['coord'])
        self.pointer = self.nframes
        self.type_count = np.array([(self.type == i).sum() for i in range(max(self.type)+1)])
        self.ntypes = len(self.type_count)
        self.valid_types = np.arange(self.ntypes)
        self.nsel = params.get('atomic_sel', None)
        if self.nsel is not None:
            self.nsel = [i for i in self.nsel if i in range(self.ntypes)]
        if any(['atomic' in l for l in labels]):
            self.nlabels = sum(self.type_count[self.nsel])
        else:
            self.nlabels = self.natoms
        perm = self.type.argsort(kind='stable')
        for l in labels:
            if l in ['coord', 'force']:
                self.data[l] = self.data[l].reshape(self.data[l].shape[0], -1, 3)[:, perm]
            if 'atomic' in l:
                try:
                    self.data[l] = self.data[l].reshape(self.data[l].shape[0], self.nlabels, -1)
                    assert self.data[l].shape[2] in (3, 9)
                except:
                    raise ValueError('Atomic label must have 3 (vector) or 9 (3x3 tensor) components per atom.')
                sel_type = self.type[np.in1d(self.type, self.nsel)]
                self.data[l] = self.data[l][:, sel_type.argsort(kind='stable')]
        self.data['box'] = self.data['box'].reshape(-1, 3, 3)
        self.data['coord'] = np.array(vmap(shift)(self.data['coord'], self.data['box']))
        self.type = self.type[perm]
        if paths is not None:
            print('# Dataset loaded: %d frames/%d atoms. Path:' % (self.nframes, self.natoms),
                  ''.join(['\n# \t\'%s\'' % abspath(path) for path in paths]))

    def _finalize_non_leaf(self):
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

    def _init_from_extxyz(self, paths, labels, params):
        '''
            Parse extxyz files, build global Z->type map (or use supplied chemical_types),
            group frames by type_count, and construct in-memory leaf subsets under a non-leaf parent.
        '''
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

        if self.chemical_types is None:
            chemical_types = tuple(sorted(all_zs))
        else:
            chemical_types = self.chemical_types
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
            natoms = len(types)
            perm = types.argsort(kind='stable')
            types_sorted = types[perm]
            permuted = {}
            for k, v in entry.items():
                if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == natoms:
                    permuted[k] = v[perm]
                else:
                    permuted[k] = v
            tc = tuple(int((types_sorted == i).sum()) for i in range(ntypes))
            grp = groups.setdefault(tc, {'type': types_sorted, 'frames': []})
            grp['frames'].append(permuted)

        subsets = []
        for tc, grp in groups.items():
            frames = grp['frames']
            data = {l: np.stack([f[l] for f in frames]) for l in labels}
            leaf = Dataset(
                paths=None, labels=labels, params=params,
                chemical_types=chemical_types,
                _in_memory={'type': grp['type'], 'data': data},
            )
            subsets.append(leaf)

        self.is_leaf = False
        self.subsets = subsets
        self._finalize_non_leaf()
        print('# Dataset loaded (extxyz): %d frames in %d composition group(s). Path:'
              % (len(raw_frames), len(subsets)),
              ''.join(['\n# \t\'%s\'' % abspath(p) for p in paths]))

    def count_max(self):
        if self.is_leaf:
            return np.array(self.type_count)
        else:
            return np.array([subset.count_max() for subset in self.subsets]).max(0)

    def fill_type(self, ntypes):
        if not self.is_leaf:
            [subset.fill_type(ntypes) for subset in self.subsets]
        else:
            self.type_count = np.pad(self.type_count, (0,ntypes-self.ntypes))

    def _get_stats(self, rcut, bs):
        if self.is_leaf:
            batch = self.get_batch(bs)[0]
            coord, box = batch['coord'], batch['box']
            r_Bnm = vmap(get_relative_coord,(0,0,None,None))(coord, box, self.type_count, self.lattice_args)[1]
            sr_BnM = [sr(jnp.concatenate(r,axis=-1), rcut) for r in r_Bnm]
            sr_sum = np.array([sr.sum() for sr in sr_BnM])
            sr_sum2 = np.array([(sr**2).sum() for sr in sr_BnM])
            sr_count = np.array([(sr>1e-15).sum() for sr in sr_BnM])
            Nnbrs = (np.concatenate(sr_BnM, axis=1) > 0).sum(2).mean() + 1
            return np.array([sr_sum, sr_sum2, sr_count, Nnbrs*np.ones_like(sr_sum)]) # (4, ntypes)
        else:
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
        if not self.is_leaf:
            subset = np.random.choice(len(self.subsets), p=self.prob)
            return self.subsets[subset].get_batch(batch_size, type)
        else:
            if type == 'label':
                batch_size = int(batch_size / self.nlabels + 1)
            if self.pointer + batch_size > self.nframes:
                self.pointer = 0
                perm = np.random.permutation(self.nframes)
                self.data = {l: self.data[l][perm] for l in self.data}
            batch = {'atomic' if 'atomic' in l else l:
                     self.data[l][self.pointer:min(self.pointer+batch_size,self.nframes)] for l in self.data}
            self.pointer += batch_size
            return batch, tuple(self.type_count), self.lattice_args

    def compute_lattice_candidate(self, rcut): # computes candidate lattice vectors within rcut for neighbor images
        if not self.is_leaf:
            for subset in self.subsets:
                subset.compute_lattice_candidate(rcut)
        else:
            self.lattice_args = compute_lattice_candidate(self.data['box'], rcut)

    def fit_energy(self):
        energy_stats = self._get_energy_stats()
        type_count, energy_mean = [np.array(x) for x in zip(*energy_stats)]
        type_count = type_count[:,self.valid_types]
        return np.linalg.lstsq(type_count, energy_mean, rcond=1e-3)[0].astype(np.float32)

    def get_atomic_label_scale(self):
        if not self.is_leaf:
            return (np.array([subset.get_atomic_label_scale() for subset in self.subsets]) * np.array(self.prob)).sum()
        else:
            label = [label for label in self.data.keys() if 'atomic' in label][0]
            return np.std(self.data[label])

    def _get_energy_stats(self):
        if self.is_leaf:
            return [(self.type_count, self.data['energy'].mean())]
        else:
            return sum([subset._get_energy_stats() for subset in self.subsets], [])

    def get_flattened_data(self):
        if self.is_leaf:
            return [{'data':self.data, 'type_count':self.type_count, 'lattice_args':self.lattice_args}]
        else:
            return sum([subset.get_flattened_data() for subset in self.subsets], [])

    def get_leaves(self):
        if self.is_leaf:
            return [self]
        return sum([s.get_leaves() for s in self.subsets], [])


def compute_lattice_candidate(boxes, rcut, print_info=True, disable_ortho=False): # boxes (nframes,3,3)
    N = 2  # This algorithm is heuristic and subject to change. Increase N in case of missing neighbors.
    ortho = not vmap(lambda box: box - jnp.diag(jnp.diag(box)))(boxes).any()
    recp_norm = jnp.linalg.norm((jnp.linalg.inv(boxes)), axis=-1)    # (nframes,3)
    n = np.ceil(rcut * recp_norm - 0.5).astype(int).max(0)           # (3,)
    lattice_cand = jnp.stack(np.meshgrid(range(-n[0],n[0]+1),range(-n[1],n[1]+1),range(-n[2],n[2]+1),indexing='ij'),axis=-1).reshape(-1,3)
    trial_points = jnp.stack(np.meshgrid(np.arange(-N,N+1),np.arange(-N,N+1),np.arange(-N,N+1)),axis=-1).reshape(-1,3) / (2*N)
    is_neighbor = jnp.linalg.norm((lattice_cand[:,None]-trial_points)[None] @ boxes[:,None], axis=-1) < rcut  # (nframes,l,t)
    lattice_cand = np.array(lattice_cand[is_neighbor.any((0,2))])
    lattice_max = is_neighbor.sum(1).max().item()
    if print_info:
        print('# Lattice vectors for neighbor images: Max %d out of %d candidates.' % (lattice_max, len(lattice_cand)))
    return {'lattice_cand': tuple(map(tuple, lattice_cand)),
            'lattice_max': lattice_max,
            'ortho': ortho if not disable_ortho else False}
