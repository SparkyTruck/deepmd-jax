import numpy as np
import jax.numpy as jnp
from glob import glob
from .utils import shift

class DataSystem():
    def __init__(self, path, labels):
        self.type = np.genfromtxt(path + '/type.raw', dtype=int)
        self.type_map = np.genfromtxt(path + '/type_map.raw', dtype=str)
        self.data = {l: np.concatenate([np.load(set+l+'.npy') for set in glob(path+'/set.*/')]) for l in labels}
        self.natoms = len(self.type)
        self.nframes = len(self.data['coord'])
        self.pointer = self.nframes
        self.type_count = np.array([(self.type == i).sum() for i in range(len(self.type_map))])
        self.type_index = np.cumsum([0] + list(self.type_count))
        for l in ['coord', 'force']:
            self.data[l] = self.data[l].reshape(-1,self.natoms,3).transpose(0,2,1)
            self.data[l] = np.concatenate([self.data[l][:,:,self.type==i] for i in range(len(self.type_map))], axis=-1)
        self.data['box'] = self.data['box'].reshape(-1,3,3).transpose(0,2,1)
        self.data['coord'] = np.array(shift(self.data['coord'], self.data['box']))
        print('Loaded data from \'%s\'' % path, 'with', self.nframes, 'frames and', self.natoms, 'atoms.')
    
    def compute_lattice_candidate(self, rcut): # computes candidate lattice vectors within rcut
        recp_norm = jnp.linalg.norm((jnp.linalg.inv(self.data['box'])), axis=1) # (nframes, 3)
        n = jnp.ceil(rcut * recp_norm).astype(int).max(0) # (3,)
        lattice_cand = jnp.array(np.meshgrid(range(-n[0],n[0]+1),range(-n[1],n[1]+1),range(-n[2],n[2]+1))).reshape(3,-1)
        lattice_vertex = jnp.array(np.meshgrid([0,1],[0,1],[0,1])).reshape(3,-1)
        cand = self.data['box'].transpose(0,2,1) @ lattice_cand[None] # (nframes, 3, -1)
        vertex = self.data['box'].transpose(0,2,1) @ lattice_vertex[None] # (nframes, 3, 8)
        self.lattice_cand = np.array(lattice_cand[:,(jnp.linalg.norm(cand[...,None]-vertex[:,:,None],axis=1).min(2)<rcut).any(0)]) # (3, -1)
        self.lattice_max = (jnp.linalg.norm(cand[...,None]-self.data['coord'][:,:,None],axis=1) < rcut).sum(1).max().item()

    def get_batch(self, batch_size):
        if self.pointer + batch_size > self.nframes:
            self.pointer = 0
            perm = np.random.permutation(self.nframes)
            self.data = {l: self.data[l][perm] for l in self.data}
        batch = {l: self.data[l][self.pointer:self.pointer+batch_size] for l in self.data}
        self.pointer += batch_size
        return batch, {'lattice_cand': tuple(map(tuple, self.lattice_cand)), 'lattice_max': self.lattice_max}
    
    def compute_Ebias(self):
        return np.linalg.lstsq(self.type_count[None], [self.data['energy'].mean()], rcond=1e-3)[0]
        



    