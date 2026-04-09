import numpy as np
import jax.numpy as jnp
from jax import vmap
from glob import glob
from os.path import abspath
from ase.io import read
from .utils import shift, get_relative_coord, sr

class DPDataset():
    def __init__(self, paths, labels, params={}):
        if type(paths[0]) == list:
            self.is_leaf = False
            self.subsets = [DPDataset(path, labels, params) for path in paths]
            self.nframes = sum([subset.nframes for subset in self.subsets])
            self.ntypes = max([subset.ntypes for subset in self.subsets])
            [subset.fill_type(self.ntypes) for subset in self.subsets]
            self.prob = np.array([subset.nframes for subset in self.subsets]) / self.nframes
            self.type_count = self.count_max()
            self.valid_types = np.arange(self.ntypes)[self.type_count > 0]
        else:
            self.is_leaf = True
            self.type = np.genfromtxt(paths[0] + '/type.raw').astype(int)
            self.data = {l: np.concatenate(sum([[np.load(set+l+'.npy') for set in sorted(glob(path+'/set.*/'))]
                                                for path in paths], [])) for l in labels}
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
            for l in labels:
                if l in ['coord', 'force']:
                    self.data[l] = self.data[l].reshape(self.data[l].shape[0],-1,3)
                    self.data[l] = self.data[l][:,self.type.argsort(kind='stable')]
                if 'atomic' in l:
                    try:
                        self.data[l] = self.data[l].reshape(self.data[l].shape[0],self.nlabels,-1)
                        assert self.data[l].shape[2] in (3, 9)
                    except:
                        raise ValueError('Atomic label must have 3 (vector) or 9 (3x3 tensor) components per atom.')
                    sel_type = self.type[np.in1d(self.type, self.nsel)]
                    self.data[l] = self.data[l][:,sel_type.argsort(kind='stable')]
            self.data['box'] = self.data['box'].reshape(-1,3,3)
            self.data['coord'] = np.array(vmap(shift)(self.data['coord'], self.data['box']))
            print('# Dataset loaded: %d frames/%d atoms. Path:'%(self.nframes,self.natoms),
                  ''.join(['\n# \t\'%s\'' % abspath(path) for path in paths]))
            
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

class EXTXYZDataset():
    def __init__(self, paths, labels, params={}):
        if type(paths[0]) == list:
            self.is_leaf = False
            self.subsets = [EXTXYZDataset(path, labels, params) for path in paths]
            self.nframes = sum([subset.nframes for subset in self.subsets])
            self.ntypes = max([subset.ntypes for subset in self.subsets])
            [subset.fill_type(self.ntypes) for subset in self.subsets]
            self.prob = np.array([subset.nframes for subset in self.subsets]) / self.nframes
            self.type_count = self.count_max()
            self.valid_types = np.arange(self.ntypes)[self.type_count > 0]
        else:
            self.is_leaf = True
            self.labels = labels
            self.params = params
            self.frames = []
            self.paths = paths
            for path in paths:
                atoms_list = read(path, index=':')
                if not isinstance(atoms_list, list):
                    atoms_list = [atoms_list]
                for atoms in atoms_list:
                    frame = {}
                    frame['box'] = np.asarray(atoms.get_cell().array, dtype=np.float32)
                    frame['coord'] = np.asarray(atoms.get_positions(), dtype=np.float32)
                    frame['coord'] = np.array(shift(frame['coord'], frame['box']))
                    frame['atomic_numbers'] = np.asarray(atoms.get_atomic_numbers(), dtype=int)
                    if 'force' in labels:
                        try:
                            frame['force'] = np.asarray(atoms.get_forces(), dtype=np.float32)
                        except RuntimeError:
                            raise ValueError('Extended xyz file missing forces for frame in %s' % path)
                    if 'energy' in labels:
                        try:
                            energy = atoms.get_potential_energy()
                            if energy is None:
                                raise ValueError('Extended xyz file missing energy for frame in %s' % path)
                            frame['energy'] = np.asarray(energy, dtype=np.float32)
                        except RuntimeError:
                            raise ValueError('Extended xyz file missing energy for frame in %s' % path)
                    for l in labels:
                        if 'atomic' in l and l not in frame:
                            if l in atoms.arrays:
                                frame[l] = np.asarray(atoms.arrays[l], dtype=np.float32)
                            else:
                                raise ValueError('Atomic label %s not found in frame arrays for %s' % (l, path))
                    if 'cell' in labels and 'cell' not in frame:
                        frame['cell'] = frame['box']
                    self.frames.append(frame)
            # Map atomic numbers to types
            all_atomic_numbers = set()
            for frame in self.frames:
                all_atomic_numbers.update(frame['atomic_numbers'])
            unique_atomic_numbers = sorted(all_atomic_numbers)
            self.atomic_to_type = {an: i for i, an in enumerate(unique_atomic_numbers)}
            self.ntypes = len(unique_atomic_numbers)
            self.valid_types = np.arange(self.ntypes)
            for frame in self.frames:
                frame['type'] = np.array([self.atomic_to_type[an] for an in frame['atomic_numbers']], dtype=int)
                # --- sort atoms by type (same as DPDataset) ---
                perm = frame['type'].argsort(kind='stable')
                frame['type'] = frame['type'][perm]
                frame['coord'] = frame['coord'][perm]
                if 'force' in frame:
                    frame['force'] = frame['force'][perm]
                # reorder atomic labels
                #for l in frame:
                #    if 'atomic' in l:
                #        frame[l] = frame[l][perm]
            self.nframes = len(self.frames)
            self.order = np.arange(self.nframes)
            self.pointer = self.nframes
            self.natoms = max([frame['coord'].shape[0] for frame in self.frames]) if self.frames else 0
            print('# EXTXYZ Dataset loaded: %d frames. Path:' % self.nframes,
                  ''.join(['\n# \t\'%s\'' % abspath(path) for path in paths]))

    def count_max(self):
        if self.is_leaf:
            type_counts = []
            for frame in self.frames:
                tc = np.array([(frame['type'] == i).sum() for i in range(self.ntypes)])
                type_counts.append(tc)
            return np.array(type_counts).max(0)
        else:
            return np.array([subset.count_max() for subset in self.subsets]).max(0)

    def fill_type(self, ntypes):
        if not self.is_leaf:
            [subset.fill_type(ntypes) for subset in self.subsets]
        else:
            self.ntypes = ntypes
            self.valid_types = np.arange(self.ntypes)

    def get_batch(self, batch_size, type='frame'):
        if not self.is_leaf:
            subset = np.random.choice(len(self.subsets), p=self.prob)
            return self.subsets[subset].get_batch(batch_size, type)
        else:
            batch_size = 1
            if self.pointer + batch_size > self.nframes:
                self.pointer = 0
                np.random.shuffle(self.order)
            index = self.order[self.pointer]
            self.pointer += batch_size
            batch = {k: v.copy() if isinstance(v, np.ndarray) else v
                     for k, v in self.frames[index].items()}
            batch['coord'] = batch['coord'][None]
            batch['box'] = batch['box'][None]
            if 'force' in batch:
                batch['force'] = batch['force'][None]
            if 'energy' in batch:
                batch['energy'] = np.array([batch['energy']])
            for l in list(batch.keys()):
                if 'atomic' in l:
                    batch[l] = batch[l][None]
            # Compute type_count for the single frame based on mapped types
            types = batch['type']
            type_count = np.array([(types == i).sum() for i in range(self.ntypes)])
            return batch, tuple(type_count), self.lattice_args

    def compute_lattice_candidate(self, rcut):
        if not self.is_leaf:
            for subset in self.subsets:
                subset.compute_lattice_candidate(rcut)
        else:
            boxes = np.stack([frame['box'] for frame in self.frames])
            self.lattice_args = compute_lattice_candidate(boxes, rcut)

    def get_flattened_data(self):
        if self.is_leaf:
            return [{'data':frame, 'lattice_args':self.lattice_args} for frame in self.frames]
        else:
            return sum([subset.get_flattened_data() for subset in self.subsets], [])

    def fit_energy(self):
        energy_stats = self._get_energy_stats()
        type_counts, energies = [np.array(x) for x in zip(*energy_stats)]
        return np.linalg.lstsq(type_counts, energies, rcond=1e-3)[0].astype(np.float32)
    
    def get_atomic_label_scale(self):
        if not self.is_leaf:
            return (np.array([subset.get_atomic_label_scale() for subset in self.subsets]) * np.array(self.prob)).sum()
        else:
            label = [l for l in self.frames[0].keys() if 'atomic' in l][0]
            all_labels = np.concatenate([frame[label].flatten() for frame in self.frames])
            return np.std(all_labels)

    def _get_energy_stats(self):
        if self.is_leaf:
            return [(np.array([(frame['type'] == i).sum() for i in range(self.ntypes)]), frame['energy']) for frame in self.frames]
        else:
            return sum([subset._get_energy_stats() for subset in self.subsets], [])

    def _get_stats(self, rcut, bs):
        if self.is_leaf:
            batch, type_count, lattice_args = self.get_batch(bs)
            coord, box = batch['coord'], batch['box']
            type_count = np.array(type_count)
            r_Bnm = vmap(get_relative_coord,(0,0,None,None))(coord, box, type_count, lattice_args)[1]
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
        return self.params


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