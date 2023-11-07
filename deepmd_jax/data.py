import numpy as np
import jax.numpy as jnp
from jax import vmap
from glob import glob
from os.path import abspath
from .utils import shift

class DPDataset():
    def __init__(self, paths, labels, params={}):
        if type(paths[0]) == list:
            self.is_leaf = False
            self.subsets = [DPDataset(path, labels, params) for path in paths]
            self.nframes = sum([subset.nframes for subset in self.subsets])
            self.prob = np.array([subset.nframes for subset in self.subsets]) / self.nframes
        else:
            self.is_leaf = True
            self.type = np.genfromtxt(paths[0] + '/type.raw').astype(int)
            self.data = {l: np.concatenate(sum([[np.load(set+l+'.npy') for set in glob(path+'/set.*/')]
                                                for path in paths], [])) for l in labels}
            self.natoms = len(self.type)
            self.nframes = len(self.data['coord'])
            self.pointer = self.nframes
            self.type_count = np.array([(self.type == i).sum() for i in range(max(self.type)+1)])
            self.type_idx = np.cumsum([0] + list(self.type_count))
            for l in labels:
                if l in ['coord', 'force']:
                    self.data[l] = self.data[l].reshape(self.data[l].shape[0],-1,3).transpose(0,2,1)
                    self.data[l] = np.concatenate([self.data[l][:,:,self.type==i] for i in range(max(self.type)+1)], axis=-1)
                if 'atomic' in l:
                    self.data[l] = self.data[l].reshape(self.data[l].shape[0],-1,3).transpose(0,2,1)
                    sel_type = self.type[np.in1d(self.type, params['atomic_sel'])]
                    self.data[l] = np.concatenate([self.data[l][:,:,sel_type==i]
                                                   for i in params['atomic_sel']], axis=-1)
            self.data['box'] = self.data['box'].reshape(-1,3,3).transpose(0,2,1)
            self.data['coord'] = np.array(vmap(shift)(self.data['coord'], self.data['box']))
            print('# Dataset loaded: %d frames/%d atoms. Path:'%(self.nframes,self.natoms),
                  ''.join(['\n# \t\'%s\'' % abspath(path) for path in paths]))

    def get_batch(self, batch_size):
        if not self.is_leaf:
            subset = np.random.choice(len(self.subsets), p=self.prob)
            return self.subsets[subset].get_batch(batch_size)
        else:
            if self.pointer + batch_size > self.nframes:
                self.pointer = 0
                perm = np.random.permutation(self.nframes)
                self.data = {l: self.data[l][perm] for l in self.data}
            batch = {'atomic' if 'atomic' in l else l:
                     self.data[l][self.pointer:self.pointer+batch_size] for l in self.data}
            self.pointer += batch_size
            return batch, tuple(self.type_idx), self.lattice_args

    def compute_lattice_candidate(self, rcut): # computes candidate lattice vectors within rcut for neighbor images
        if not self.is_leaf:
            for subset in self.subsets:
                subset.compute_lattice_candidate(rcut)
        else:
            self.lattice_args = compute_lattice_candidate(self.data['box'], rcut, True)
    
    def fit_energy(self):
        energy_stats = self._get_energy_stats()
        type_count, energy_mean = [np.array(x) for x in zip(*energy_stats)]
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

def compute_lattice_candidate(boxes, rcut, print_message=False):
    # input boxes (nframes, 3, 3)
    ortho = not (boxes - vmap(jnp.diag)(vmap(jnp.diag)(boxes))).any()
    recp_norm = jnp.linalg.norm((jnp.linalg.inv(boxes)), axis=1) # (nframes, 3)
    n = jnp.ceil(rcut * recp_norm - 0.5).astype(int).max(0) # (3,)
    lattice_cand = jnp.array(np.meshgrid(range(-n[0],n[0]+1),range(-n[1],n[1]+1),range(-n[2],n[2]+1))).reshape(3,-1)
    lattice_vertex = jnp.array(np.meshgrid([-0.5,0.5],[-0.5,0.5],[-0.5,0.5])).reshape(3,-1)
    lattice_vertex = jnp.concatenate([lattice_vertex, jnp.zeros((3,1))], axis=1)
    cand = boxes @ lattice_cand[None] # (nframes, 3, -1)
    N = 2 # This algorithm is heuristic and subject to change. Increase N in case of missing neighbors.
    trial_points = boxes @ jnp.array(np.meshgrid(np.arange(-N,N+1),np.arange(-N,N+1),np.arange(-N,N+1))).reshape(3,-1) / (2*N)
    lattice_cand = np.array(lattice_cand[:,(jnp.linalg.norm(cand[...,None]-trial_points[:,:,None],axis=1).min(2)<rcut).any(0)]) # (3, -1)
    lattice_max = (jnp.linalg.norm(cand[...,None]-trial_points[:,:,None],axis=1) < rcut).sum(1).max().item()
    if print_message:
        print('# Lattice vectors for neighbor images: Max %d out of %d condidates.' % (lattice_max, lattice_cand.shape[1]))
    return {'lattice_cand': tuple(map(tuple, lattice_cand)), 'lattice_max': lattice_max, 'ortho': ortho}