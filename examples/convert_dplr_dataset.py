# Used in DPLR; convert energy-force dataset to short range dataset by subtracting long range part; 
# This code typically runs within a minute on a GPU
# Input a list of source datasets, output a single target dataset
# Source datasets should have the same type index; if not, run the code for each group of sets with the same type index
deepmd_jax_path    = '../'              # Path to deepmd_jax package; change if you run this script at a different directory
precision          = 'default'          # 'default'(fp32), 'low'(tf32), 'high'(fp64)
source_paths       = ['data/chunyi_dplr/data/energy_force_data/data/data' + str(i) for i in range(1,46)] \
                   + ['data/chunyi_dplr/data/energy_force_data/data/data_ex' + str(i) for i in range(1,7)]
target_path        = 'data/chunyi_dplr/data/energy_force_data/data_sr'
wannier_model_path = 'trained_models/dw_chunyidplr_1.pkl'
q_atoms            = [6, 1]             # charge of atomic cores, here Oxygen and Hydrogen
q_wc               = [-8]               # charge of wannier center/centroid
atomic_sel         = [0]                # type of wannier center/centroid association, here only Oxygen
beta               = 0.4                # inverse spread of the point charge distribution
resolution         = 0.2                # grid length of particle mesh = resolution / beta

import numpy as np
import jax.numpy as jnp
from jax import jit, value_and_grad
import jax, sys, os, datetime, pickle
import flax.linen as nn
from functools import partial
sys.path.append(os.path.abspath(deepmd_jax_path))
from deepmd_jax.data import DPDataset
from deepmd_jax.dpmodel import DPModel
from deepmd_jax.utils import get_p3mlr, get_p3mlr_grid_size
if precision == 'default':
    jax.config.update('jax_default_matmul_precision', 'float32')
if precision == 'high':
    jax.config.update('jax_enable_x64', True)
np.set_printoptions(precision=4, suppress=True)
print('# Program start at', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'on device:', jax.devices()[:1])

with open(wannier_model_path, 'rb') as file:
    m = pickle.load(file)
model, variables = m['model'], m['variables']
dataset = DPDataset(source_paths, ['coord','box','force','energy'])
dataset.compute_lattice_candidate(model.params['rcut'])
dataset.pointer = 0
batch, type_idx, lattice_args = dataset.get_batch(1)
static_args = nn.FrozenDict({'type_idx':type_idx, 'lattice':lattice_args})
pred_fn = jit(model.apply, static_argnums=(3,))
qatoms = jnp.concatenate([jnp.ones(type_idx[i+1]-type_idx[i])*q_atoms[i] for i in range(len(type_idx)-1)])
qwc = jnp.concatenate([jnp.ones(type_idx[atomic_sel[i]+1]-type_idx[atomic_sel[i]])*q_wc[i] for i in range(len(atomic_sel))])
q = jnp.concatenate([qatoms, qwc])
sel_idx = np.concatenate([(i in atomic_sel)*np.ones(type_idx[i+1]-type_idx[i],dtype=bool) for i in range(len(type_idx)-1)])
def lr_energy(coord, box, M):
    wc_rel = pred_fn(variables, coord, box, static_args)[0]
    wc = coord.T[sel_idx] + wc_rel
    coord_and_wc = jnp.concatenate([coord.T, wc])
    p3mlr = get_p3mlr(jnp.diag(box), beta, M)
    return p3mlr(coord_and_wc, q), wc
@partial(jit, static_argnums=(2,))
def lr_energy_and_force(coord, box, M):
    (e, wc), negf = value_and_grad(lr_energy, has_aux=True)(coord, box, M)
    return e, -negf, wc
dataset.pointer = 0
energy_lr, force_lr, wc_pred = [], [], []
for i in range(dataset.nframes):
    M = get_p3mlr_grid_size(np.diag(dataset.data['box'][i]), beta, resolution=0.2)
    e_lr, f_lr, wc = lr_energy_and_force(dataset.data['coord'][i], dataset.data['box'][i], M)
    energy_lr.append(e_lr), force_lr.append(f_lr), wc_pred.append(wc)
energy_lr, force_lr, wc = np.array(energy_lr), np.array(force_lr).transpose(0,2,1), np.array(wc_pred)
energy_full, force_full = dataset.data['energy'], dataset.data['force'].transpose(0,2,1)
energy_sr, force_sr = energy_full - energy_lr, force_full - force_lr

# create dataset at target path
os.makedirs(target_path, exist_ok=True)
os.makedirs(target_path + '/set.000', exist_ok=True)
type_array = np.concatenate([np.ones(type_idx[i+1]-type_idx[i],dtype=int)*i for i in range(len(type_idx)-1)])
np.savetxt(target_path + '/type.raw', type_array, fmt='%d')
np.save(target_path + '/set.000/coord.npy', dataset.data['coord'].transpose(0,2,1).reshape(dataset.nframes,-1))
np.save(target_path + '/set.000/box.npy', dataset.data['box'].transpose(0,2,1).reshape(dataset.nframes,-1))
np.save(target_path + '/set.000/energy.npy', energy_sr)
np.save(target_path + '/set.000/force.npy', force_sr.reshape(dataset.nframes,-1))
print('Saved short range dataset to', target_path)

# Print mean/std for sanity check
print('Energy (mean,std): Full (%.2f, %.2f) LR (%.2f, %.2f) SR (%.2f, %.2f) Ratio of std %.3f'
    % (energy_full.mean(), energy_full.std(), energy_lr.mean(), energy_lr.std(),
       energy_sr.mean(), energy_sr.std(), energy_sr.std()/energy_full.std()))
print('Force (mean,std): Full (%.2f, %.2f) LR (%.2f, %.2f) SR (%.2f, %.2f) Ratio of std %.3f'
    % (force_full.mean(), force_full.std(), force_lr.mean(), force_lr.std(),
         force_sr.mean(), force_sr.std(), force_sr.std()/force_full.std()))
