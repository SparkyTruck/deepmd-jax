# Used in DPLR; convert energy-force dataset to short range dataset by subtracting long range part; 
# This code typically runs within a minute on a GPU
# Input a list of source datasets, output a single target dataset
# Source datasets should have the same type index; if not, run the code for each group of sets with the same type index
precision          = 'default'          # 'default'(fp32), 'low'(mixed 32-16), 'high'(fp64)
source_paths       = ['source_dataset_path'] # Path to the source data files
target_path        = 'target_dataset_path'   # Path to the target data file
wannier_model_path = 'model.pkl'        # Path to the model file
q_atoms            = [6, 1]             # charge of atomic cores, here Oxygen and Hydrogen
q_wc               = [-8]               # charge of wannier center/centroid
atomic_sel         = [0]                # type of wannier center/centroid association, here only Oxygen
beta               = 0.4                # inverse spread of the point charge distribution
resolution         = 0.2                # particle mesh grid length = resolution / beta

import numpy as np
import jax.numpy as jnp
from jax import jit, value_and_grad
import jax, datetime
import flax.linen as nn
from functools import partial
from deepmd_jax import data, utils
from deepmd_jax.dpmodel import DPModel
if precision == 'default':
    jax.config.update('jax_default_matmul_precision', 'float32')
if precision == 'high':
    jax.config.update('jax_enable_x64', True)
np.set_printoptions(precision=4, suppress=True)
print('# Program start at', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'on device:', jax.devices()[:1])
model, variables = utils.load_model(wannier_model_path)
dataset = data.DPDataset(source_paths, ['coord','box','force','energy'])
dataset.compute_lattice_candidate(model.params['rcut'])
dataset.pointer = 0
batch, type_count, lattice_args = dataset.get_batch(1)
sel_type_count = tuple(np.array(type_count)[model.params['nsel']])
static_args = nn.FrozenDict({'type_count':type_count, 'lattice':lattice_args})
pred_fn = jit(model.apply, static_argnums=(3,))
qatoms = np.repeat(q_atoms, type_count)
qwc = np.repeat(q_wc, sel_type_count)
def lr_energy(coord, box, M):
    wc = model.wc_predict(variables, coord, box, static_args)
    p3mlr_fn = utils.get_p3mlr_fn(jnp.diag(box), beta, M)
    return p3mlr_fn(jnp.concatenate([coord, wc]), jnp.concatenate([qatoms, qwc])), wc
@partial(jit, static_argnums=(2,))
def lr_energy_and_force(coord, box, M):
    (e, wc), negf = value_and_grad(lr_energy, has_aux=True)(coord, box, M)
    return e, -negf, wc
dataset.pointer = 0
energy_lr, force_lr, wc = [], [], []
for i in range(dataset.nframes):
    M = utils.get_p3mlr_grid_size(np.diag(dataset.data['box'][i]), beta, resolution=resolution)
    e_lr, f_lr, wc_pred = lr_energy_and_force(dataset.data['coord'][i], dataset.data['box'][i], M)
    energy_lr.append(e_lr), force_lr.append(f_lr), wc.append(wc_pred)
energy_lr, force_lr, wc = np.array(energy_lr), np.array(force_lr), np.array(wc)
energy_full, force_full = dataset.data['energy'], dataset.data['force']
energy_sr, force_sr = energy_full - energy_lr, force_full - force_lr
utils.save_dataset(target_path, {'coord':dataset.data['coord'],
                                 'box':dataset.data['box'],
                                 'energy':energy_sr,
                                 'force':force_sr,
                                 'type': np.repeat(range(len(type_count)),type_count)})
# Print mean/std for sanity check
print('Energy (mean,std): Full (%.2f, %.2f) LR (%.2f, %.2f) SR (%.2f, %.2f) Ratio of std %.3f'
    % (energy_full.mean(), energy_full.std(), energy_lr.mean(), energy_lr.std(),
       energy_sr.mean(), energy_sr.std(), energy_sr.std()/energy_full.std()))
print('Force (mean,std): Full (%.2f, %.2f) LR (%.2f, %.2f) SR (%.2f, %.2f) Ratio of std %.3f'
    % (force_full.mean(), force_full.std(), force_lr.mean(), force_lr.std(),
         force_sr.mean(), force_sr.std(), force_sr.std()/force_full.std()))
