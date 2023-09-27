import numpy as np
import jax.numpy as jnp
from jax import vmap
from glob import glob
from os.path import abspath
from .utils import shift

class SingleDataSystem():
    def __init__(self, paths, labels):
        self.type = np.genfromtxt(paths[0] + '/type.raw', dtype=int)
        self.type_map = np.genfromtxt(paths[0] + '/type_map.raw', dtype=str)
        self.data = {l: np.concatenate(sum([[np.load(set+l+'.npy') for set in glob(path+'/set.*/')]
                                             for path in paths], [])) for l in labels}
        self.natoms = len(self.type)
        self.nframes = len(self.data['coord'])
        self.pointer = self.nframes
        self.type_count = np.array([(self.type == i).sum() for i in range(len(self.type_map))])
        self.type_index = np.cumsum([0] + list(self.type_count))
        for l in ['coord', 'force']:
            self.data[l] = self.data[l].reshape(-1,self.natoms,3).transpose(0,2,1)
            self.data[l] = np.concatenate([self.data[l][:,:,self.type==i] for i in range(len(self.type_map))], axis=-1)
        self.data['box'] = self.data['box'].reshape(-1,3,3).transpose(0,2,1)
        self.ortho = not (self.data['box'] - vmap(jnp.diag)(vmap(jnp.diag)(self.data['box']))).any()
        self.data['coord'] = np.array(vmap(shift,(0,0,None))(self.data['coord'], self.data['box'], self.ortho))
        print('SingleDataSystem loaded from: \n', ''.join(['\'%s\'\n' % abspath(path) for path in paths]),
              'with', self.nframes, 'frames and', self.natoms, 'atoms per frame.')
    
    def compute_lattice_candidate(self, rcut): # computes candidate lattice vectors within rcut for neighbor images
        recp_norm = jnp.linalg.norm((jnp.linalg.inv(self.data['box'])), axis=1) # (nframes, 3)
        n = jnp.ceil(rcut * recp_norm - 0.5).astype(int).max(0) # (3,)
        lattice_cand = jnp.array(np.meshgrid(range(-n[0],n[0]+1),range(-n[1],n[1]+1),range(-n[2],n[2]+1))).reshape(3,-1)
        lattice_vertex = jnp.array(np.meshgrid([-0.5,0.5],[-0.5,0.5],[-0.5,0.5])).reshape(3,-1)
        lattice_vertex = jnp.concatenate([lattice_vertex, jnp.zeros((3,1))], axis=1)
        cand = self.data['box'] @ lattice_cand[None] # (nframes, 3, -1)
        self.lattice_cand = np.array(lattice_cand[:,(jnp.linalg.norm(cand[...,None]-self.data['coord'][:,:,None],axis=1).min(2)<rcut).any(0)]) # (3, -1)
        self.lattice_max = (jnp.linalg.norm(cand[...,None]-self.data['coord'][:,:,None],axis=1) < rcut).sum(1).max().item()
        print('Lattice vectors computed with %d neighbor image condidates and max %d images.' % (self.lattice_cand.shape[1], self.lattice_max))

    def get_batch(self, batch_size):
        if self.pointer + batch_size > self.nframes:
            self.pointer = 0
            perm = np.random.permutation(self.nframes)
            self.data = {l: self.data[l][perm] for l in self.data}
        batch = {l: self.data[l][self.pointer:self.pointer+batch_size] for l in self.data}
        self.pointer += batch_size
        return batch, {'lattice_cand': tuple(map(tuple, self.lattice_cand)), 'lattice_max': self.lattice_max, 'ortho': self.ortho}
    
    def compute_Ebias(self):
        return np.linalg.lstsq(self.type_count[None], [self.data['energy'].mean()], rcond=1e-3)[0]
        



    