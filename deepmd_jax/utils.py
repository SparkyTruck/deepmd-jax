import jax.numpy as jnp
from jax import lax
import numpy as np
import flax.linen as nn

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

def slice_type(array, type_index, axis): # slice array by atom type into list of subarrays
    return [lax.slice_in_dim(array, type_index[i], type_index[i+1], axis=axis)
            for i in range(len(type_index)-1)]

def get_relative_coord(coord_3N, box_33, lattice_args):
    lattice_cand = np.array(lattice_args['lattice_cand'])
    N, X, Y = coord_3N.shape[1], lattice_cand.shape[1], lattice_args['lattice_max']
    x_3NN = shift((coord_3N[:,None] - coord_3N[:,:,None]).reshape(3,-1), box_33, lattice_args['ortho']).reshape(3,N,N)
    x_NNX3 = (x_3NN[:,:,:,None] - (box_33 @ lattice_cand)[:,None,None]).transpose(1,2,3,0)
    r_NNX = jnp.linalg.norm(x_NNX3 + 1e-15*jnp.eye(N)[:,:,None,None], axis=-1)
    if X == Y:
        x_3NM = x_NNX3.reshape(N,-1,3).transpose(2,0,1)
    else:
        if Y == 1:
            idx_NNY = r_NNX.argmin(axis=-1, keepdims=True)
        else:
            idx_NNY = r_NNX.argpartition(lattice_args['lattice_max'], axis=-1)[:,:,:lattice_args['lattice_max']]
        x_3NM = jnp.take_along_axis(x_NNX3, idx_NNY[...,None], axis=2).reshape(N,-1,3).transpose(2,0,1)
    r_NM = jnp.linalg.norm(x_3NM + (1e-15*jnp.eye(N)[...,None]*jnp.ones((N,N,Y))).reshape(1,N,-1), axis=0)
    return x_3NM, r_NM

he_init = nn.initializers.he_normal()
vs_init = nn.initializers.variance_scaling(0.5, "fan_avg", "normal")
std_init = nn.initializers.normal(1)
dt_init = lambda k, s: 0.1 + nn.initializers.normal(0.001)(k, s)
zero_init = nn.initializers.zeros_init()

class embedding_net(nn.Module):
    widths: list
    @nn.compact
    def __call__(self, x):
        for i in range(len(self.widths)):
            if i == 0:
                x = nn.tanh(nn.Dense(self.widths[i], kernel_init=he_init, bias_init=std_init)(x))
            else:
                K = self.widths[i] / self.widths[i-1]
                x_prev = (x[...,None] * jnp.ones((int(K),))).reshape((x.shape[:-1]) + (-1,))
                # x_prev = (x[...,None,:] * jnp.ones((int(K),1))).reshape((x.shape[:-1]) + (-1,))
                x = nn.tanh(nn.Dense(self.widths[i], kernel_init=he_init, bias_init=std_init)(x))
                if K.is_integer():
                    x += x_prev
        return x

class fitting_net(nn.Module):
    widths: list
    @nn.compact
    def __call__(self, x):
        for i in range(len(self.widths)):
            x_prev = x
            x = nn.tanh(nn.Dense(self.widths[i], kernel_init=vs_init, bias_init=std_init)(x))
            if i > 0 and self.widths[i] == self.widths[i-1]:
                dt = self.param('dt'+str(i), dt_init, (self.widths[i],))
                x = x * dt + x_prev 
        x = nn.Dense(1, bias_init=nn.initializers.normal())(x)
        return x