import jax.numpy as jnp
import jax
from jax import lax
import numpy as np
import flax.linen as nn
from functools import partial

def shift(coord, box, ortho=False): # shift coordinates to the parallelepiped around the origin
    if ortho:
        box = jnp.diag(box)[:,None]
        return (coord - box/2) % box - box/2
    else:
        relcoord = jnp.linalg.solve(box, coord)
        relcoord = (relcoord - 0.5) % 1 - 0.5
        return box @ relcoord

def sr(r, rcut): # 1/r with smooth cutoff at rcut
    t = r / rcut
    return (r>1e-14) * (t<1) / (r+1e-15) * (1-3*t**2+2*t**3)

def split(array, type_idx, axis=0, K=1): # split array by idx into list of subarrays with device count K
    if K == 1:
        return [lax.slice_in_dim(array, type_idx[i], type_idx[i+1], axis=axis) for i in range(len(type_idx)-1)]
        # return jnp.split(array, type_idx[1:-1], axis)
    if axis < 0:
        axis += len(array.shape)
    array = array.reshape(array.shape[:axis] + (K,-1) + array.shape[axis+1:])
    return [x.reshape(x.shape[:axis]+(-1,)+x.shape[axis+2:]) for x in jnp.split(array, type_idx[1:-1], axis=axis+1)]

def concat(array_list, axis=0, K=1): # concatenate subarray into list of arrays with device count K
    if K == 1:
        return jnp.concatenate(array_list, axis=axis)
    if axis < 0:
        axis += len(array_list[0].shape)
    return jnp.concatenate([array.reshape(array.shape[:axis]+(K,-1)+array.shape[axis+1:]) for array in array_list],
                           axis=axis+1).reshape(array_list[0].shape[:axis]+(-1,)+array_list[0].shape[axis+1:])

def tensor_3to6(x, axis, bias=0):
    return jnp.concatenate([x**2-bias, 2**0.5 * x * jnp.roll(x,shift=1,axis=axis)], axis=axis)

def get_relative_coord(coord_3N, box_33, lattice_args, nbr_idx=None):
    if nbr_idx is None:
        lattice_cand = np.array(lattice_args['lattice_cand'])
        N, X, Y = coord_3N.shape[1], lattice_cand.shape[1], lattice_args['lattice_max']
        x_3NN = shift((coord_3N[:,None] - coord_3N[:,:,None]).reshape(3,-1), box_33, lattice_args['ortho']).reshape(3,N,N)
        if X == Y:
            x_3NM = (x_3NN[...,None] - (box_33 @ lattice_cand)[:,None,None,:]).reshape(3,N,-1)
        else:
            x_3NNX = (x_3NN[:,:,:,None] - (box_33 @ lattice_cand)[:,None,None])
            r_NNX = jnp.linalg.norm(x_3NNX + 1e-15*jnp.eye(N)[None,:,:,None], axis=0)
            if Y == 1:
                idx_NNY = r_NNX.argmin(axis=-1, keepdims=True)
            else:
                idx_NNY = r_NNX.argpartition(lattice_args['lattice_max'], axis=-1)[:,:,:lattice_args['lattice_max']]
            x_3NM = jnp.take_along_axis(x_3NNX, idx_NNY[None], axis=-1).reshape(3,N,-1)
        r_NM = jnp.linalg.norm(x_3NM + (1e-15*jnp.eye(N)[...,None]*jnp.ones((N,N,Y),dtype=jnp.float32)).reshape(1,N,-1), axis=0)
    else:
        x_3NM = shift((coord_3N[:,nbr_idx] - coord_3N[:,:,None]).reshape(3,-1), box_33, True).reshape(3,nbr_idx.shape[0],-1)
        r_NM = jnp.linalg.norm(jnp.where(jnp.abs(x_3NM) > 1e-15, x_3NM, 1e-15), axis=0) * (nbr_idx < nbr_idx.shape[0])
    return x_3NM, r_NM

he_init = nn.initializers.he_normal()
original_init = nn.initializers.variance_scaling(0.5, "fan_avg", "truncated_normal")
std_init = jax.nn.initializers.truncated_normal(1)
embed_dt_init = lambda k, s: 0.5 + nn.initializers.normal(0.01)(k, s)
fit_dt_init = lambda k, s: 0.1 + nn.initializers.normal(0.001)(k, s)
linear_init = nn.initializers.variance_scaling(0.05, "fan_in", "truncated_normal")
ones_init = nn.initializers.ones_init()
zeros_init = nn.initializers.zeros_init()
    
class embedding_net(nn.Module):
    widths: list
    in_bias_only: bool = False
    out_linear_only: bool = False
    dt_layers: tuple = ()
    @nn.compact
    def __call__(self, x):
        for i in range(len(self.widths)):
            if i == 0 and self.in_bias_only:
                x = nn.tanh(x + self.param('bias',zeros_init,(self.widths[0],)))
            elif i == 0:
                x = nn.tanh(nn.Dense(self.widths[i], kernel_init=he_init, bias_init=std_init)(x))
            else:
                Z = self.widths[i] / self.widths[i-1]
                assert Z.is_integer()
                if self.out_linear_only and i == len(self.widths) - 1:
                    x = jnp.tile(nn.Dense(self.widths[i-1], kernel_init=linear_init, use_bias=False)(x), int(Z))
                else:
                    x_prev = x
                    x = nn.tanh(nn.Dense(self.widths[i], kernel_init=he_init, bias_init=std_init)(x))
                    if i in self.dt_layers:
                        x = x * self.param('dt'+str(i), embed_dt_init, (self.widths[i],)) + jnp.tile(x_prev, int(Z))
                    else:
                        x += jnp.tile(x_prev, int(Z))
        return x

class fitting_net(nn.Module):
    widths: list
    @nn.compact
    def __call__(self, x):
        for i in range(len(self.widths)):
            x_prev = x
            x = nn.tanh(nn.Dense(self.widths[i], kernel_init=original_init, bias_init=std_init)(x))
            if i > 0 and self.widths[i] == self.widths[i-1]:
                dt = self.param('dt'+str(i), fit_dt_init, (self.widths[i],))
                x = x * dt + x_prev 
        x = nn.Dense(1, bias_init=zeros_init)(x)
        return x

class linear_norm(nn.Module):
    width: int
    @nn.compact
    def __call__(self, x):
        return (nn.Dense(self.width, kernel_init=linear_init, use_bias=False)(x) * self.param('norm',ones_init,(1,)))
    
def get_mask_by_device(type_idx, K):
    mask = jnp.concatenate([jnp.concatenate([c, jnp.zeros((-c.shape[0]%K,),dtype=bool)]).reshape(K,-1)
                            for c in split(jnp.ones((type_idx[-1],),dtype=bool),type_idx,0)], axis=1).reshape(-1)
    return lax.with_sharding_constraint(mask, jax.sharding.PositionalSharding(jax.devices()))

@partial(jax.jit, static_argnums=(1,2))
def reorder_by_device(coord, type_idx, K): # Fill coord with zeros to make Natoms divisible by device count K
    return jnp.concatenate([jnp.concatenate([c, jnp.zeros((-c.shape[0]%K,3),dtype=c.dtype)]).reshape(K,-1,3)
                            for c in split(coord,type_idx,0)], axis=1).reshape(-1, 3)