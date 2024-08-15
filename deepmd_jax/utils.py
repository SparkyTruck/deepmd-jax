import jax.numpy as jnp
import jax
from jax import lax, vmap, jacfwd
import numpy as np
import flax.linen as nn
import pickle, os
from scipy.interpolate import PPoly, BPoly

def shift(coord, box, ortho=False): # shift coordinates to the parallelepiped around the origin
    if ortho:
        box = jnp.diag(box)
        return (coord - box/2) % box - box/2
    else:
        fractional_coord = coord @ jnp.linalg.inv(box)
        fractional_coord = (fractional_coord - 0.5) % 1 - 0.5
        return fractional_coord @ box

def sr(r, rcut): # 1/r with smooth cutoff at rcut
    t = r / rcut
    return (r>1e-14) * (t<1) / (r+1e-15) * (1-3*t**2+2*t**3)

def split(array, type_count, axis=0, K=1): # split array by idx into list of subarrays with device count K
    type_idx = np.cumsum(list(type_count))
    if K == 1:
        return jnp.split(array, type_idx[:-1], axis)
    if axis < 0:
        axis += len(array.shape)
    array = array.reshape(array.shape[:axis] + (K,-1) + array.shape[axis+1:])
    return [x.reshape(x.shape[:axis]+(-1,)+x.shape[axis+2:]) for x in jnp.split(array, type_idx[:-1], axis=axis+1)]

def concat(array_list, axis=0, K=1): # concatenate subarray into list of arrays with device count K
    if K == 1:
        return jnp.concatenate(array_list, axis=axis)
    if axis < 0:
        axis += len(array_list[0].shape)
    return jnp.concatenate([array.reshape(array.shape[:axis]+(K,-1)+array.shape[axis+1:]) for array in array_list],
                           axis=axis+1).reshape(array_list[0].shape[:axis]+(-1,)+array_list[0].shape[axis+1:])

def tensor_3to6(x, axis, bias=0):
    return jnp.concatenate([x**2-bias, 2**0.5 * x * jnp.roll(x,shift=1,axis=axis)], axis=axis)

def get_relative_coord(coord_N3, box_33, type_count, lattice_args, nbrs_nm=None):
    x_n3m, r_nm = [], []
    coord_n3 = split(coord_N3, type_count, K=(1 if nbrs_nm is None else jax.device_count()))
    for i in range(len(type_count)):
        x, r = [], []
        for j in range(len(type_count)):
            N, M = len(coord_n3[i]), len(coord_n3[j])
            if N * M == 0:
                M = M*lattice_args['lattice_max'] if nbrs_nm is None else nbrs_nm[i][j].shape[1]
                x_N3M = jnp.ones((N, 3, M), dtype=coord_N3.dtype)
                r_NM = jnp.ones((N, M), dtype=coord_N3.dtype)
            else:
                if nbrs_nm is None:
                    lattice_cand = jnp.array(lattice_args['lattice_cand'])
                    X, Y = len(lattice_cand), lattice_args['lattice_max']
                    x_N3M = shift(coord_n3[j] - coord_n3[i][:,None], box_33, lattice_args['ortho']).transpose(0,2,1)
                    if X > 1:
                        x_N3MX = x_N3M[...,None] - (lattice_cand @ box_33).T[:,None]
                        if X == Y:
                            x_N3M = x_N3MX.reshape(N,3,-1)
                        else:
                            r_NMX = jnp.linalg.norm(jnp.where(jnp.abs(x_N3MX) > 1e-15, x_N3MX, 1e-15), axis=1)
                            idx_NMY = r_NMX.argpartition(lattice_args['lattice_max'], axis=-1)[:,:,:lattice_args['lattice_max']]
                            x_N3M = jnp.take_along_axis(x_N3MX, idx_NMY[:,None], axis=-1).reshape(N,3,-1)
                    r_NM = jnp.linalg.norm(jnp.where(jnp.abs(x_N3M) > 1e-15, x_N3M, 1e-15), axis=1)
                else:
                    x_N3M = shift(coord_n3[j][nbrs_nm[i][j]] - coord_n3[i][:,None], box_33, True).transpose(0,2,1)
                    r_NM = jnp.linalg.norm(jnp.where(jnp.abs(x_N3M) > 1e-15, x_N3M, 1e-15), axis=1) * (nbrs_nm[i][j] < len(coord_n3[j]))
            x.append(x_N3M)
            r.append(r_NM)
        x_n3m.append(x)
        r_nm.append(r)
    return x_n3m, r_nm

he_init = nn.initializers.he_normal()
original_init = nn.initializers.variance_scaling(0.5, "fan_avg", "truncated_normal")
std_init = jax.nn.initializers.truncated_normal(1)
embed_dt_init = lambda k, s: 0.5 + nn.initializers.normal(0.01)(k, s)
fit_dt_init = lambda k, s: 0.1 + nn.initializers.normal(0.001)(k, s)
linear_init = nn.initializers.variance_scaling(0.05, "fan_in", "truncated_normal")
ones_init = nn.initializers.ones_init()
zeros_init = nn.initializers.zeros_init()

MEM_CAP = None
    
class embedding_net(nn.Module):
    widths: list
    in_bias_only: bool = False
    out_linear_only: bool = False
    dt_layers: tuple = ()
    @nn.compact
    def __call__(self, x, compress=False, reducer=None):
        if not compress:
            for i in range(len(self.widths)):
                if i == 0 and self.in_bias_only:
                    x = nn.tanh(x + self.param('bias',zeros_init,(self.widths[0],)))
                elif i == 0:
                    x = nn.tanh(nn.Dense(self.widths[i], kernel_init=he_init, bias_init=std_init)(x))
                else:
                    Z = self.widths[i] / self.widths[i-1]
                    assert Z.is_integer() or (1/Z).is_integer()
                    x_prev = jnp.repeat(x, int(Z), axis=-1) if Z.is_integer() else x.reshape(x.shape[:-1]+(int(1/Z),int(x.shape[-1]*Z))).mean(-2)
                    if self.out_linear_only and i == len(self.widths) - 1:
                        x = jnp.repeat(nn.Dense(self.widths[i-1], kernel_init=linear_init, use_bias=False)(x), int(Z), axis=-1)
                    else:
                        x = nn.tanh(nn.Dense(self.widths[i], kernel_init=he_init, bias_init=std_init)(x))
                        if i in self.dt_layers:
                            x = x * self.param('dt'+str(i), embed_dt_init, (self.widths[i],)) + x_prev
                        else:
                            x += x_prev
            return x if reducer is None else reducer @ x
        else:
            srmin = self.variable('compress_var', 'srmin').value
            srmax = self.variable('compress_var', 'srmax').value
            poly_coeff = self.variable('compress_var', 'poly_coeff').value
            Ngrids = poly_coeff.shape[0]
            delta = (srmax - srmin) / Ngrids
            x = x[...,0] - srmin
            idx = (x / delta).astype(int)
            x0 = x - idx * delta
            if MEM_CAP is not None:
                Nchunks = int(poly_coeff.shape[-1] * x.nbytes / (jax.device_count() * MEM_CAP) + 1)
                print('# Low memory mode enabled with Nchunks =', Nchunks)
                pad = -len(idx) % Nchunks
                idx = jnp.pad(idx,((0,pad),(0,0))).reshape(Nchunks, -1, idx.shape[1])
                x0 = jnp.pad(x0,((0,pad),(0,0))).reshape(Nchunks, -1, x0.shape[1])
                reducer = jnp.pad(reducer,((0,pad),(0,0),(0,0))).reshape(Nchunks, -1, reducer.shape[1], reducer.shape[2])
                def body(inputs):
                    idx, x0, reducer = inputs
                    coeff = poly_coeff[idx]
                    embed = sum([coeff[...,i,:] * x0[...,None]**(5-i) for i in range(6)])
                    return reducer @ embed
                return lax.map(body, [idx,x0,reducer]).reshape(-1,reducer.shape[-2],poly_coeff.shape[-1])[:x.shape[0]]
            else:
                coeff = poly_coeff[idx]
                embed = sum([coeff[...,i,:] * x0[...,None]**(5-i) for i in range(6)])
                return reducer @ embed if reducer is not None else embed    

class fitting_net(nn.Module):
    widths: list
    use_final: bool = True
    @nn.compact
    def __call__(self, x):
        for i in range(len(self.widths)):
            x_prev = x
            x = nn.tanh(nn.Dense(self.widths[i], kernel_init=original_init, bias_init=std_init)(x))
            if i > 0 and self.widths[i] == self.widths[i-1]:
                dt = self.param('dt'+str(i), fit_dt_init, (self.widths[i],))
                x = x * dt + x_prev 
        if self.use_final:
            x = nn.Dense(1, bias_init=zeros_init)(x)
        return x

class linear_norm(nn.Module):
    width: int
    @nn.compact
    def __call__(self, x):
        return (nn.Dense(self.width, kernel_init=linear_init, use_bias=False)(x) * self.param('norm',ones_init,(1,)))
    
def get_mask_by_device(type_count):
    K = jax.device_count()
    mask = concat([concat([jnp.ones(count, dtype=bool), jnp.zeros((-count%K,), dtype=bool)]).reshape(K,-1)
                   for count in type_count], axis=1).reshape(-1)
    return lax.with_sharding_constraint(mask, jax.sharding.PositionalSharding(jax.devices()))

def reorder_by_device(coord, type_count): # Pad with zeros in first dimension to be divisible by device count K
    K = jax.device_count()
    coord = jnp.concatenate([jnp.pad(c, ((0,-c.shape[0]%K),)+((0,0),)*(c.ndim-1)).reshape(K,-1,*c.shape[1:])
                            for c in split(coord,type_count)], axis=1).reshape(-1, *coord.shape[1:])
    sharding = jax.sharding.PositionalSharding(jax.devices())
    return lax.with_sharding_constraint(coord,sharding.replicate())

def get_p3mlr_grid_size(box3, beta, resolution=0.2): # resolution=0.1 for better accuracy
    M = tuple((box3*beta/resolution).astype(int))
    return M

def get_p3mlr_fn(box3, beta, M=None, resolution=0.2): # PPPM long range with TSC assignment
    if M is None:
        M = get_p3mlr_grid_size(box3, beta, resolution)
    K = jax.device_count()
    sharding = jax.sharding.PositionalSharding(jax.devices()).reshape(K,1)
    cube_idx = (jnp.stack(jnp.meshgrid(*([jnp.array([-1,0,1])]*3),indexing='ij'))).reshape(3,27)
    MM = jnp.array(M).reshape(3,1,1,1)
    kgrid = jnp.stack(jnp.meshgrid(*[jnp.arange(m) for m in M], indexing='ij'))
    kgrid = 2*jnp.pi/box3[:,None,None,None] * ((kgrid-MM/2)%MM-MM/2)
    ksquare = (kgrid ** 2).sum(0)
    z = kgrid * (box3/jnp.array(M))[:,None,None,None] / 2
    sinz = jnp.sin(z)
    w3k = jnp.prod(jnp.where(z==0, 1, (sinz/z)**3), axis=0)
    Sk = (jnp.prod(1 - sinz**2 + 2/15*sinz**4, axis=0))**2
    kfactor = (14.399645*2*jnp.pi/jnp.prod(box3)) * jnp.exp(-ksquare/(4*beta**2)) * w3k**2/(Sk*ksquare)
    kfactor = kfactor.at[0,0,0].set(0.)
    def assign_to_grid(coord_N3, q_N): 
        grid = jnp.zeros(M)
        M3 = jnp.array(M)
        coord_3N = ((coord_N3 % box3) / box3 * M3).T
        center_idx_3N = jnp.rint(coord_3N).astype(int) # in [0, M3]
        r_3N = coord_3N - center_idx_3N # lies in (-0.5, 0.5)
        fr_33N = jnp.stack([(r_3N-0.5)**2/2,
                            0.75 - r_3N**2,
                            (r_3N+0.5)**2/2]) # TSC assignment
        fr_27N = (fr_33N[:,None,None,0,:]*fr_33N[:,None,1,:]*fr_33N[:,2,:]).reshape(27,-1)
        all_idx = (center_idx_3N[:,None] + cube_idx[:,:,None]).reshape(3,-1) % M3[:,None]
        grid = grid.at[tuple(all_idx)].add((q_N*fr_27N).reshape(-1))
        return grid
    def p3mlr_fn(coord_N3, q_N): # coord in Angstrom, q in elementary charge, returns energy in eV
        if K > 1:
            coord_N3 = lax.with_sharding_constraint(reorder_by_device(coord_N3,(0,len(q_N))), sharding)
            q_N = lax.with_sharding_constraint(reorder_by_device(q_N,(0,len(q_N))), sharding.reshape(K))
        grid = assign_to_grid(coord_N3, q_N)
        skfactor = jnp.fft.fftn(grid)
        return (kfactor * (skfactor*skfactor.conj()).real).sum()
    return p3mlr_fn

def save_model(path, model, variables):
    with open(path, 'wb') as file:
        pickle.dump({'model':model, 'variables':variables}, file)
    print('# Model saved to \'%s\'.' % path)

def load_model(path):
    sharding = jax.sharding.PositionalSharding(jax.devices()).replicate()
    with open(path, 'rb') as file:
        m = pickle.load(file)
    print('# Model loaded from \'%s\'.' % path)
    return m['model'], jax.device_put(m['variables'], sharding)

def save_dataset(target_path, labeled_data):
    os.makedirs(target_path, exist_ok=True)
    os.makedirs(target_path + '/set.000', exist_ok=True)
    N = labeled_data['coord'].shape[0]
    np.savetxt(target_path + '/type.raw', labeled_data['type'], fmt='%d')
    for l in labeled_data:
        if l == 'type': continue
        labeled_data[l] = labeled_data[l].reshape(N,) if labeled_data[l].size == N else labeled_data[l].reshape(N,-1)
        np.save(target_path + '/set.000/' + l + '.npy', labeled_data[l])
    print('Saved dataset with %d frames to' % N, target_path)

def compress_model(model, variables, Ngrids, rmin):
    model.params['is_compressed'] = True
    nsel = [list(model.params['valid_types']).index(i) for i in model.params['nsel']] if model.params['atomic'] else None
    Y = len(model.params['sr_mean'])
    Z = len(nsel) if model.params['atomic'] else Y
    # prepare list of names, widths, sr_ranges (srmin and srmax) for all compressible embedding nets
    sr_range = (np.array([0.,sr(rmin,model.params['rcut'])])-np.array(model.params['sr_mean'])[:,None])/np.array(model.params['sr_std'])[:,None]
    if model.params['use_mp']:
        names = ['embedding_net_%d' % i for i in range(Y*Z + 2*Y**2)]
        widths = Y*Z*[model.params['embed_widths']+model.params['embedMP_widths'][:1]] + 2*Y**2*[model.params['embed_widths']]
        sr_ranges = np.concatenate([np.repeat(sr_range[nsel] if model.params['atomic'] else sr_range, Y, axis=0),
                                    np.tile(np.repeat(sr_range, Y, axis=0),(2,1))])
        out_linear_onlys = Y*Z*[True] + 2*Y**2*[False]
    else:
        names = ['embedding_net_%d' % i for i in range(Y*Z)]
        widths = Y*Z * [model.params['embed_widths']]
        sr_ranges = np.repeat(sr_range[nsel] if model.params['atomic'] else sr_range, Y, axis=0)
        out_linear_onlys = Y*Z * [False]
    # compute poly_coeff for each embedding net
    variables['compress_var'] = {}
    err0, err1, err2 = [], [], []
    x64_is_enabled = jax.config.read('jax_enable_x64')
    for name, width, sr_range, out_linear_only in zip(names, widths, sr_ranges, out_linear_onlys):
        net = embedding_net(width, out_linear_only=out_linear_only, name=name)
        var = {'params': variables['params'][name]}
        jax.config.update("jax_enable_x64", True)
        r = np.linspace(sr_range[0], sr_range[1], Ngrids+1)
        f0 = net.apply(var, r[:,None])
        f1 = vmap(jax.jacfwd(lambda x: net.apply(var, x[None])))(r)
        f2 = vmap(jax.jacfwd(jax.jacfwd(lambda x: net.apply(var, x[None]))))(r)
        poly = PPoly.from_bernstein_basis(BPoly.from_derivatives(r, np.stack([f0,f1,f2], axis=1), orders=None))
        if not x64_is_enabled:
            jax.config.update("jax_enable_x64", False)
        variables['compress_var'][name] = {}
        variables['compress_var'][name]['srmin'] = jnp.array(sr_range[0])
        variables['compress_var'][name]['srmax'] = jnp.array(sr_range[1])
        variables['compress_var'][name]['poly_coeff'] = jnp.array(poly.c.transpose(1,0,2))
        # validate the compression error
        newvar = {'compress_var': variables['compress_var'][name]}
        r = np.linspace(sr_range[0], sr_range[1]*(1-1e-6), (Ngrids*10+1))
        err0.append(net.apply(var,r[...,None]) - net.apply(newvar,r[...,None],True))
        err1.append(vmap(jacfwd(lambda r: net.apply(var,r[...,None]) - net.apply(newvar,r[...,None],True)))(r))
        err2.append(vmap(jacfwd(jacfwd(lambda r: net.apply(var,r[...,None]) - net.apply(newvar,r[...,None],True))))(r))
    err0, err1, err2 = jnp.concatenate(err0,axis=None), jnp.concatenate(err1,axis=None), jnp.concatenate(err2,axis=None)
    print('# Model Compressed: %d embedding nets, %d intervals' % (len(names), Ngrids))
    print('# Compression (0,1,2)-order error: Mean = (%.2e,%.2e,%.2e), Max = (%.2e,%.2e,%.2e)'
            % (err0.mean(), err1.mean(), err2.mean(), err0.max(), err1.max(), err2.max()))
    return model, variables
