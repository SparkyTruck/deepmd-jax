deepmd_jax_path  = '../'           # Path to deepmd_jax package; change if you run this script at a different directory
precision        = 'default'       # 'default'(fp32), 'low'(mixed 32-16), 'high'(fp64)
model_path       = 'model.pkl'     # Path to the model file
data_paths       = ['data/water_128'] # Path to the data files
test_batch_size  = 8               # Batch size for testing

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
import jax, sys, os, datetime
import flax.linen as nn
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(deepmd_jax_path))
from deepmd_jax import data, utils
from deepmd_jax.dpmodel import DPModel
if precision == 'default':
    jax.config.update('jax_default_matmul_precision', 'float32')
if precision == 'high':
    jax.config.update('jax_enable_x64', True)
np.set_printoptions(precision=4, suppress=True)
print('# Program start at', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'on device:', jax.devices()[:1])

model, variables = utils.load_model(model_path)
dataset = data.DPDataset(data_paths, ['coord', 'box', 'force', 'energy'])
dataset.compute_lattice_candidate(model.params['rcut'])
force_true, force_pred, energy_true, energy_pred = [], [], [], []
e_and_f = jit(vmap(model.energy_and_force, in_axes=(None,0,0,None)), static_argnums=(3,))
dataset.pointer = 0
for i in range(dataset.nframes//test_batch_size):
    batch, type_count, lattice_args = dataset.get_batch(test_batch_size)
    static_args = nn.FrozenDict({'type_count':type_count, 'lattice': lattice_args})
    force_true.append(batch['force'])
    energy_true.append(batch['energy'])
    e, f = e_and_f(variables, batch['coord'], batch['box'], static_args)
    force_pred.append(np.array(f))
    energy_pred.append(np.array(e))
force_true = np.concatenate(force_true, axis=0)
force_pred = np.concatenate(force_pred, axis=0)
energy_true = np.concatenate(energy_true, axis=0)
energy_pred = np.concatenate(energy_pred, axis=0)
force_err = force_true - force_pred
energy_err = (energy_true - energy_pred)
print('Mean force error = %.4f' % (force_err**2).mean()**0.5)
print('Mean energy error = %.4f, Mean energy shift = %.4f' % (energy_err.std(), energy_err.mean()))
# plt.plot(force_true.flatten(), force_pred.flatten(), '.') # parity plot for force
# plt.savefig('force_parity.png')
plt.figure()
plt.plot(energy_true, energy_pred, '.') # parity plot for energy
plt.savefig('energy_parity.png')
plt.figure()
plt.plot(np.abs(force_err).max(1).flatten(), '.') # max error for each atom's force
plt.savefig('force_max_error.png')