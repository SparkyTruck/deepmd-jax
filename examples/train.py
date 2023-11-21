# Config parameters
deepmd_jax_path = '../'           # Path to deepmd_jax package; change if you run this script at a different directory
precision       = 'default'       # 'default'(fp32), 'low'(mixed 32-16), 'high'(fp64)
save_name       = 'trained_models/new_dpmp_water32.pkl' # model save path
model_type      = 'energy'        # 'energy' or 'atomic' (e.g. wannier)
atomic_sel      = [0]             # select atom type for prediction (only for 'atomic' model)
atomic_label    = 'atomic_dipole' # data file prefix for 'atomic' model; string must contain 'atomic'

# Dataset in DeepMD-kit format; nested paths like [[dat1,dat2],[dat3]] allowed
# Note: Here the atomic type index of dat1,dat2 must be the same, but that of dat3 can be different
# train_paths     = ['data/chunyi_dplr/data/dipole_data']
train_paths     = ['data/water_128_shifted/']
# train_paths     = ['data/chunyi_dplr/data/energy_force_data/data_sr']
# train_paths     = ['data/chunyi_dplr/data/energy_force_data/data/data' + str(i) for i in range(1,46)] \
#                    + ['data/chunyi_dplr/data/energy_force_data/data/data_ex' + str(i) for i in range(1,7)]
use_val_data    = False           # if not, next line is ignored
val_paths       = ['/pscratch/sd/r/ruiqig/polaron_cp2k/aimd/polaron_full_val']

# Model parameters
rcut            = 6.0             # cutoff radius (Angstrom)
use_2nd_tensor  = True            # Use 2nd order tensor descriptor for more accuracy, slightly slower
use_mp          = True            # Use message passing (DP-MP) model for even more accuracy (slower) 
compress        = False            # Compress model after training for faster inference. Rec: True  
embed_widths    = [32,32,64]      # Rec: [32,32,64] (for accuracy try [48,48,96])
embedMP_widths  = [32,64]         # Rec: [32,64]; Only used in MP; (Try [64,64] or [96,96] according to embed_widths)
fit_widths      = [64,64,64]      # For 'atomic' model, fit_widths[-1] must equal embed_widths[-1](DP)/embedMP_widths[-1](DP-MP)
axis_neurons    = 12              # Rec: 8-16

# Training parameters
batch_size      = 1               # training batch size; Rec: 128 <= labels_per_frame*batch_size <= 512
val_batch_size  = 8               # validation batch size. Too much can cause OOM error.
lr              = 0.002           # learning rate at start. Rec: 0.001/0.002 for 'energy', 0.01 for 'atomic'
s_pref_e        = 0.02            # starting prefactor for energy loss
l_pref_e        = 1               # limit prefactor for energy loss, increase for energy accuracy
s_pref_f        = 1000            # starting prefactor for force loss
l_pref_f        = 1               # limit prefactor for force loss, increase for force accuracy
total_steps     = 200000          # total training steps. Rec: 1e6 for 'energy', 1e5 for 'atomic'
print_every     = 1000            # for printing loss and validation

# parameters you usually don't need to change
lr_limit        = 5e-7            # learning rate at end of training
compress_Ngrids = 512             # Number of intervals used in compression
compress_rmin   = 0.6             # Lower bound for interatomic distance in compression
beta2           = 0.99            # adam optimizer parameter
l_smoothing     = 20              # smoothing factor for loss printing
decay_steps     = 5000            # learning rate exponentially decays every decay_steps
getstat_bs      = 64              # batch size for computing model statistics at initialization

# From here on you don't need to change anything unless you know what you are doing
import numpy as np
from jax import jit, random, tree_util
import jax, optax, sys, os, datetime
import flax.linen as nn
from time import time
from functools import partial
sys.path.append(os.path.abspath(deepmd_jax_path))
from deepmd_jax import data, utils
from deepmd_jax.dpmodel import DPModel, compress_model
if precision == 'default':
    jax.config.update('jax_default_matmul_precision', 'float32')
if precision == 'high':
    jax.config.update('jax_enable_x64', True)
TIC = time()
print('# Program start at', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'on device:', jax.devices()[:1])

labels = ['coord','box'] + (['force','energy'] if model_type == 'energy' else [atomic_label])
train_data     = data.DPDataset(train_paths, labels, {'atomic_sel':atomic_sel})
val_data       = data.DPDataset(val_paths, labels, {'atomic_sel':atomic_sel}) if use_val_data else None
model = DPModel({'embed_widths':embed_widths[:-1] if use_mp else embed_widths,
                 'embedMP_widths':embed_widths[-1:] + embedMP_widths if use_mp else None,
                 'fit_widths':fit_widths,
                 'axis':axis_neurons,
                 'Ebias':train_data.fit_energy() if model_type == 'energy' else None,
                 'rcut':rcut,
                 'use_2nd':use_2nd_tensor,
                 'use_mp':use_mp,
                 'atomic':True if model_type == 'atomic' else False,
                 'nsel':atomic_sel if model_type == 'atomic' else None,
                 'out_norm': 1. if model_type == 'energy' else train_data.get_atomic_label_scale()})
train_data.compute_lattice_candidate(rcut)
if use_val_data:
    val_data.compute_lattice_candidate(rcut)
batch, type_count, lattice_args = train_data.get_batch(getstat_bs)
static_args         = nn.FrozenDict({'type_count':type_count, 'lattice':lattice_args})
model.get_stats(batch['coord'], batch['box'], static_args)
print('# Model params:', model.params)
variables           = model.init(random.PRNGKey(np.random.randint(42)), batch['coord'][0], batch['box'][0], static_args)
print('# Model initialized. Precision: %s. Parameter count: %d.' % 
            ({'default': 'fp32', 'low': 'fp32-16', 'high': 'fp64'}[precision], 
            sum(i.size for i in tree_util.tree_flatten(variables)[0])))
lr_scheduler        = optax.exponential_decay(init_value=lr, transition_steps=decay_steps,
                        decay_rate=(lr_limit/lr)**(decay_steps/(total_steps-decay_steps)), transition_begin=0, staircase=True)
optimizer           = optax.adam(learning_rate=lr_scheduler, b2=beta2)
opt_state           = optimizer.init(variables)
loss, loss_and_grad = model.get_loss_fn()
print('# Optimizer initialized, lr starts from %.1e. Starting training...' % lr)

state = {'loss_avg':0., 'iteration':0} | ({} if model_type == 'atomic' else {'le_avg':0., 'lf_avg':0.})
@partial(jit, static_argnums=(4,))
def train_step(batch, variables, opt_state, state, static_args):
    r = lr_scheduler(state['iteration']) / lr
    if model_type == 'energy':
        pref = {'e': s_pref_e*r + l_pref_e*(1-r), 'f': s_pref_f*r + l_pref_f*(1-r)}
        (loss_total, (loss_e, loss_f)), grads = loss_and_grad(variables, batch, pref, static_args)
        for key, value in zip(['loss_avg', 'le_avg', 'lf_avg'], [loss_total, loss_e, loss_f]):
            state[key] = state[key] * (1-1/l_smoothing) + value
    else:
        loss_total, grads = loss_and_grad(variables, batch, static_args)
        state['loss_avg'] = state['loss_avg'] * (1-1/l_smoothing) + loss_total
    updates, opt_state = optimizer.update(grads, opt_state, variables)
    variables = optax.apply_updates(variables, updates)
    state['iteration'] += 1
    return variables, opt_state, state

@partial(jit, static_argnums=(2,))
def val_step(batch, variables, static_args):
    if model_type == 'energy':
        pref = {'e': 1, 'f': 1}
        _, (loss_e, loss_f) = loss(variables, batch, pref, static_args)
        return loss_e, loss_f
    else:
        loss_total = loss(variables, batch, static_args)
        return loss_total

tic = time()
for iteration in range(total_steps+1):
    batch, type_count, lattice_args = train_data.get_batch(batch_size)
    static_args = nn.FrozenDict({'type_count':tuple(type_count), 'lattice':lattice_args})
    variables, opt_state, state = train_step(batch, variables, opt_state, state, static_args)
    if iteration % print_every == 0:
        if use_val_data:
            val_batch, type_count, lattice_args = val_data.get_batch(val_batch_size)
            static_args = nn.FrozenDict({'type_count':tuple(type_count), 'lattice':lattice_args})
            loss_val = val_step(val_batch, variables, static_args)
        beta = l_smoothing * (1 - (1/l_smoothing)**(iteration+1))
        print('Iter %7d' % iteration
              + ' L %7.5f' % (state['loss_avg']/beta)**0.5
              + (' LE %7.5f' % (state['le_avg']/beta)**0.5 if model_type == 'energy' else '')
              + (' LF %7.5f' % (state['lf_avg']/beta)**0.5 if model_type == 'energy' else '')
              + (' LEval %7.5f' % loss_val[0]**0.5 if model_type == 'energy' and use_val_data else '')
              + (' LFval %7.5f' % loss_val[1]**0.5 if model_type == 'energy' and use_val_data else '')
              + (' Lval %7.5f' % loss_val**0.5 if model_type == 'atomic' and use_val_data else '')
              + ' Time %.2fs' % (time()-tic))
        tic = time()
if compress:
    model, variables = utils.compress_model(model, variables, compress_Ngrids, compress_rmin)
utils.save_model(save_name, model, variables)
T = int(time() - TIC)
print('# Training finished in %dh %dm %ds.' % (T//3600,(T%3600)//60,T%60))