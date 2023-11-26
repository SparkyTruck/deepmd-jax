# this module contains functions that requires jax_md
import jax.numpy as jnp
import jax
from jax import lax
import numpy as np
from jax_md import space, partition
from .utils import split, get_mask_by_device, reorder_by_device, split

def get_type_mask_fns(type_count):
    mask_fns = []
    K = jax.device_count()
    Kmask = get_mask_by_device(type_count)
    type_idx_filled_each = np.cumsum(np.concatenate([[0], -(-type_count//K)]))
    N_each = type_idx_filled_each[-1]
    for i in range(len(type_count)):
        def mask_fn(idx, i=i):
            idx = jax.device_put(idx, jax.sharding.PositionalSharding(jax.devices()).reshape(K,1))
            cond = Kmask[:,None] * ((idx%N_each >= type_idx_filled_each[i]) * (idx%N_each < type_idx_filled_each[i+1]))
            cond *= ~((idx//N_each == K-1) * (idx%N_each >= type_idx_filled_each[i+1] - (-type_count[i])%K)) 
            return jnp.where(cond, idx, N_each * K)
        mask_fns.append(mask_fn)
    return mask_fns
class NeighborList():
    def __init__(self, coord, box, type_count, rcut, dr_buffer, size):
        self.type_count, self.box = tuple(type_count), box
        self.mask_fns = get_type_mask_fns(np.array(type_count))
        self.mask_fn = lambda idx: jax.device_put(idx, jax.sharding.PositionalSharding(jax.devices()).reshape(jax.device_count(),1))
        self.rcut, self.dr_buffer, self.size = rcut, dr_buffer, size
    def allocate(self, coord):
        displace = space.periodic(self.box)[0]
        coord = reorder_by_device(coord%self.box, self.type_count)
        test_nbr = partition.neighbor_list(displace, self.box, self.rcut+self.dr_buffer,
                                               capacity_multiplier=1.).allocate(coord)
        self.knbr = np.array([int(((fn(test_nbr.idx)<len(coord)).sum(1).max())*self.size) for fn in self.mask_fns])
        self.knbr = np.where(self.knbr==0, 1, self.knbr + 1 + max(int(20*(self.size-1.2)),0))
        all_buffer = (sum(self.knbr)+1) / test_nbr.idx.shape[1]
        print('# Neighborlist allocated with size', np.array(self.knbr)-1)
        return partition.neighbor_list(displace, self.box, self.rcut, dr_threshold=self.dr_buffer, capacity_multiplier=all_buffer,
                        custom_mask_function=self.mask_fn).allocate(coord)
    def update(self, coord, nbrs):
        return nbrs.update(reorder_by_device(coord%self.box, self.type_count))
    def check_dr_overflow(self, coord, ref):
        return (jnp.linalg.norm((coord-ref-self.box/2)
                    %self.box - self.box/2, axis=-1) > self.dr_buffer/2).any()
    def get_nm(self, nbrs):
        K = jax.device_count()
        sharding = jax.sharding.PositionalSharding(jax.devices()).reshape(K, 1)
        nbr_idx = lax.with_sharding_constraint(nbrs.idx, sharding)
        nbrs_idx = [-lax.top_k(-fn(nbr_idx), self.knbr[i])[0] for i, fn in enumerate(self.mask_fns)]
        type_count_new = [-(-self.type_count[i]//K) for i in range(len(self.type_count))]
        type_idx_new = np.cumsum([0] + list(type_count_new))
        nbrs_nm = [mlist for mlist in zip(*[split(jnp.where(nbrs < type_idx_new[-1]*K,
            nbrs%type_idx_new[-1] - type_idx_new[i] + (nbrs//type_idx_new[-1]) * (type_count_new[i]),
            type_idx_new[-1]*K), type_count_new, K=K) for i, nbrs in enumerate(nbrs_idx)])]
        overflow = jnp.array([(idx.max(axis=1)<type_idx_new[-1]*K).any() for idx in nbrs_idx]).any() | nbrs.did_buffer_overflow
        return nbrs_nm, overflow