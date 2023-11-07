import jax.numpy as jnp
import jax
from jax import lax
import numpy as np
import flax.linen as nn
from jax_md import space, partition
from .utils import split, get_mask_by_device, reorder_by_device

class NeighborListLoader():
    def __init__(self, box, type_idx, rcut_all, size, K=1):
        self.type_idx, self.K, self.nbrlists = type_idx, K, []
        displace, _ = space.periodic(box)
        Kmask = get_mask_by_device(type_idx, K)
        atom_count = (np.array(type_idx[1:]) - np.array(type_idx[:-1]))
        type_idx_filled_each = np.cumsum(np.concatenate([[0], -(-atom_count//K)]))
        N_each = type_idx_filled_each[-1]
        for i in range(len(self.type_idx) - 1):
            def mask_fn(idx, i=i):
                idx = jax.device_put(idx, jax.sharding.PositionalSharding(jax.devices()).reshape(K,1))
                cond = Kmask[:,None] * ((idx%N_each >= type_idx_filled_each[i]) * (idx%N_each < type_idx_filled_each[i+1]))
                cond *= ~((idx//N_each == K-1) * (idx%N_each >= type_idx_filled_each[i+1] - (-atom_count[i])%K)) 
                return jnp.where(cond, idx, N_each * K)
            self.nbrlists.append(partition.neighbor_list(displace, box, rcut_all[i],
                capacity_multiplier=size, custom_mask_function=mask_fn))
    def allocate(self, coord):
        nbrs_list = [nbrlist.allocate(reorder_by_device(coord,tuple(self.type_idx),self.K)) for nbrlist in self.nbrlists]
        print('# Neighborlist allocated with size', [nbrs.idx.shape[1] for nbrs in nbrs_list])
        return nbrs_list
    def update(self, coord, nbrs_list):
        return [nbrlist.update(reorder_by_device(coord,tuple(self.type_idx),self.K),nbrs) for nbrlist,nbrs in zip(self.nbrlists,nbrs_list)]