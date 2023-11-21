# this module contains functions that requires jax_md
import jax.numpy as jnp
import jax
from jax import lax
import numpy as np
import flax.linen as nn
from jax_md import space, partition
from .utils import split, get_mask_by_device, reorder_by_device

class NeighborListLoader():
    def __init__(self, box, type_count, rcut_all, size):
        K = jax.device_count()
        self.type_count, self.nbrlists = type_count, []
        displace, _ = space.periodic(box)
        Kmask = get_mask_by_device(type_count)
        type_idx_filled_each = np.cumsum(np.concatenate([[0], -(-type_count//K)]))
        N_each = type_idx_filled_each[-1]
        for i in range(len(self.type_count)):
            def mask_fn(idx, i=i):
                idx = jax.device_put(idx, jax.sharding.PositionalSharding(jax.devices()).reshape(K,1))
                cond = Kmask[:,None] * ((idx%N_each >= type_idx_filled_each[i]) * (idx%N_each < type_idx_filled_each[i+1]))
                cond *= ~((idx//N_each == K-1) * (idx%N_each >= type_idx_filled_each[i+1] - (-type_count[i])%K)) 
                return jnp.where(cond, idx, N_each * K)
            self.nbrlists.append(partition.neighbor_list(displace, box, rcut_all[i],
                capacity_multiplier=size, custom_mask_function=mask_fn))
    def allocate(self, coord):
        nbrs_list = [nbrlist.allocate(reorder_by_device(coord,tuple(self.type_count))) for nbrlist in self.nbrlists]
        print('# Neighborlist allocated with size', [nbrs.idx.shape[1] for nbrs in nbrs_list])
        return nbrs_list
    def update(self, coord, nbrs_list):
        return [nbrlist.update(reorder_by_device(coord,tuple(self.type_count)),nbrs) for nbrlist,nbrs in zip(self.nbrlists,nbrs_list)]
    