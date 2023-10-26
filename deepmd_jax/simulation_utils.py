import jax.numpy as jnp
import jax
from jax import lax
import numpy as np
import flax.linen as nn
from jax_md import space, partition

class NeighborListLoader():
    def __init__(self, box, type_idx, rcut_all, size):
        self.type_idx = type_idx
        self.nbrlists = []
        displace, _ = space.periodic(box)
        for i in range(len(self.type_idx) - 1):
            mask_fn = lambda idx, i=i: jnp.where((idx >= self.type_idx[i]) * (idx < self.type_idx[i+1]), idx, self.type_idx[-1])
            self.nbrlists.append(partition.neighbor_list(displace, box, rcut_all[i],
                capacity_multiplier=size, custom_mask_function=mask_fn))
    def allocate(self, coord):
        return [nbrlist.allocate(coord) for nbrlist in self.nbrlists]
    def update(self, coord, nbrs_list):
        return [nbrlist.update(coord, nbrs) for nbrlist, nbrs in zip(self.nbrlists, nbrs_list)]