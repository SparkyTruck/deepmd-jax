import numpy as np
from jax import jit, random, tree_util
import jax, optax, sys, os, pickle, datetime
import flax.linen as nn
from time import time
from functools import partial
# Path to deepmd_jax; change if you're running this script from a different directory
sys.path.append(os.path.abspath('../'))
from deepmd_jax.data import DPDataset
from deepmd_jax.dpmodel import DPModel
print('# Program start at', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'on device:', jax.devices()[:1])
TIC = time()
# Recommended to use 32 bit, you can change to 16 (for mixed 32/16 bit) or 64 bit
precision      = '32' # '16' '32' '64'
if precision == '32':
    jax.config.update('jax_default_matmul_precision', 'float32')
if precision == '64':
    jax.config.update('jax_enable_x64', True)

# Beginning config parameters; change as you need
save_name      = 'trained_models/dp_polaron_tensor_1.pkl' # model save path
model_type     = 'energy'        # 'energy': global scalar model; 'atomic': atomic vector model e.g. wannier 
atomic_sel     = [0]             # select atom type for model predictions (only for 'atomic' model)
atomic_label   = 'atomic_dipole' # file prefix for labels in set.xxx for 'atomic' model; string should contain 'atomic'
use_val_data   = True            # use validation data or not
# Dataset in DeepMD-kit format; nested lists of paths like [[dat1,dat2],[dat3]] can be used
# In this example dat1,dat2 should have the same type index but dat3 can have different type index
train_path     = ['/pscratch/sd/r/ruiqig/polaron_cp2k/aimd/aimd-water/water_128/']
val_path       = ['/pscratch/sd/r/ruiqig/polaron_cp2k/aimd/aimd-water/water_128_val/']
rcut           = 6.0             # cutoff radius (Angstrom)
use_2nd_tensor = True            # Use 2nd order tensor descriptor for more accuracy (recommended, slightly slower)
use_mp         = False           # Use message passing (DP-MP) model for even more accuracy (slower)  
embed_widths   = [32,32,64]      # Recommended: [32,32,64] (Try [48,48,96] for accuracy)
embedMP_widths = [64,64]         # Used in MP; Recommended: [64,64] (Try [32,64] or [96,96], adjust according to embed_widths)
fit_widths     = [64,64,64]      # For 'atomic' model, assert fit_widths[-1,] == embed_widths[-1](DP) or embedMP_widths[-1](DP-MP)
axis_neurons   = 12              # Recommended: 8-16; Most of network parameters are in this layer
batch_size     = 1               # training batch size, Recommended: labels_per_frame * batch_size in (128, 512)
val_batch_size = 8               # validation batch size. Beware: too much can cause OOM error.
lr             = 0.002           # learning rate at the beginning, Recommended: 1e-3 or 2e-3 for DP/DP-MP, 1e-2 for atomic
s_pref_e       = 0.02            # starting prefactor for energy loss
l_pref_e       = 1               # limit prefactor for energy loss, increase this for more accurate energy prediction
s_pref_f       = 1000            # starting prefactor for force loss
l_pref_f       = 1               # limit prefactor for force loss, increase this for more accurate force prediction
total_steps    = 500000          # total training steps; Recommended: 5e5-1e6 for 'energy', 5e4-2e5 for 'atomic'
print_every    = 1000            # print loss and conduct validation every print_every steps in output
# parameters you usually don't need to worry about
limit_lr       = 1e-6            # learning rate at the end of training
beta2          = 0.99            # adam optimizer parameter
l_smoothing    = 20              # smoothing factor for loss display
decay_steps    = 5000            # learning rate decays every decay_steps
getstat_bs     = 64              # batch size for computing model statistics at the beginning

# From here on you don't need to change anything unless you know what you are doing
labels = ['coord','box'] + (['force','energy'] if model_type == 'energy' else [atomic_label])
train_data     = DPDataset(train_path, labels)
val_data       = DPDataset(val_path, labels) if use_val_data else None
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
batch, type_idx, lattice_args = train_data.get_batch(getstat_bs)
static_args         = nn.FrozenDict({'type_idx':type_idx, 'lattice':lattice_args})
model.get_stats(batch['coord'], batch['box'], static_args)
print('# Model statistics computed.')
variables           = model.init(random.PRNGKey(np.random.randint(42)), batch['coord'][0], batch['box'][0], static_args)
print('# \'%s\' model initialized (float%s). Parameter count: %d' %
      (model_type, precision, sum(i.size for i in tree_util.tree_flatten(variables)[0])))
lr_scheduler        = optax.exponential_decay(init_value=lr, transition_steps=decay_steps,
                        decay_rate=(limit_lr/lr)**(decay_steps/(total_steps-decay_steps)), transition_begin=0, staircase=True)
optimizer           = optax.adam(learning_rate=lr_scheduler, b2=beta2)
opt_state           = optimizer.init(variables)
loss, loss_and_grad = model.get_loss_fn()
print('# Optimizer initialized. Starting training...')

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
    batch, type_idx, lattice_args = train_data.get_batch(batch_size)
    static_args = nn.FrozenDict({'type_idx':tuple(type_idx), 'lattice':lattice_args})
    variables, opt_state, state = train_step(batch, variables, opt_state, state, static_args)
    if iteration % print_every == 0:
        if use_val_data:
            val_batch, type_idx, lattice_args = val_data.get_batch(val_batch_size)
            static_args = nn.FrozenDict({'type_idx':tuple(type_idx), 'lattice':lattice_args})
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

with open(save_name, 'wb') as file:
    pickle.dump({'model':model, 'variables':variables}, file)
T = int(time() - TIC)
print('# Model saved to \'%s\'.' % save_name)
print('# Training finished in %dh %dm %ds.' % (T//3600,(T%3600)//60,T%60))