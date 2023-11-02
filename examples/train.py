import numpy as np
from jax import jit, random, tree_util
import jax, optax, sys, os, pickle, datetime
import flax.linen as nn
from time import time
sys.path.append(os.path.abspath('../'))
from deepmd_jax.data import SingleDataSystem
from deepmd_jax.dpmodel import DPModel
print('Program start at', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'on device:', jax.devices()[:1])
TIC = time()
# Default to use 32 bit, you can change to 16 (for mixed 16/32bit) or 64 bit (not recommended)
precision      = '32' # '16' '32' '64'
if precision == '32':
    jax.config.update('jax_default_matmul_precision', 'float32')
if precision == '64':
    jax.config.update('jax_enable_x64', True)

# DP config parameters
save_name      = 'trained_models/model_water_final_2.pkl' # model save name
# train_data     = SingleDataSystem(['data/polaron_train/'], ['coord', 'box', 'force', 'energy'])
train_data     = SingleDataSystem(['/pscratch/sd/r/ruiqig/polaron_cp2k/aimd/aimd-water/water_128/'], ['coord', 'box', 'force', 'energy'])
use_val_data   = True # if False, comment next line
# val_data       = SingleDataSystem(['data/polaron_val/'], ['coord', 'box', 'force', 'energy'])
val_data       = SingleDataSystem(['/pscratch/sd/r/ruiqig/polaron_cp2k/aimd/aimd-water/water_128_val/'], ['coord', 'box', 'force', 'energy'])
model_type     = 'scalar' # 'scalar'(energy) or 'vector'(wannier centers, etc.)
rcut           = 6.0
use_2nd_tensor = False # Use 2nd order tensor descriptor for more accuracy
use_mp         = False # Use message passing (DP-MP) model for even more accuracy (slower)  
embed_widths   = [24, 48, 96]  # Recommended width for DP model
# embed_widths   = [32, 32]      # Recommended width for DP-MP model
embedMP_widths = [64, 64, 64]  # only used in DP-MP model
fit_widths     = [128, 128, 128]
axis_neurons   = 12
batch_size     = 1
val_batch_size = 8
lr             = 0.002
s_pref_e       = 0.02
l_pref_e       = 1
s_pref_f       = 1000
l_pref_f       = 10
total_steps    = 500000
decay_steps    = 5000
decay_rate     = 0.95
print_every    = 1000

# parameters you usually don't need to change
RANDOM_SEED    = np.random.randint(1000)
beta2          = 0.99
l_smoothing    = 20
getstat_bs     = 64

train_data.compute_lattice_candidate(rcut)
if use_val_data:
    val_data.compute_lattice_candidate(rcut)
model = DPModel({'embed_widths':embed_widths,
                 'embedMP_widths':embedMP_widths if use_mp else None,
                 'fit_widths':fit_widths,
                 'axis':axis_neurons,
                 'Ebias':train_data.compute_Ebias(),
                 'rcut':rcut,
                 'use_2nd':use_2nd_tensor,
                 'use_mp':use_mp,
                 'model_type':model_type,})
batch, lattice_args = train_data.get_batch(getstat_bs)
static_args = nn.FrozenDict({'lattice': lattice_args, 'type_idx':tuple(train_data.type_idx)})
model.get_stats(batch['coord'], batch['box'], static_args)
print('Model statistics computed.')
variables = model.init(random.PRNGKey(RANDOM_SEED), batch['coord'][0], batch['box'][0], static_args)
print('\'%s\' model initialized with precision \'%s\' and %d parameters.' % 
    (model_type, precision, sum(i.size for i in tree_util.tree_flatten(variables)[0])))
lr_scheduler = optax.exponential_decay(init_value=lr, transition_steps=decay_steps,
                    decay_rate=decay_rate, transition_begin=0, staircase=True)
optimizer = optax.adam(learning_rate=lr_scheduler, b2=beta2)
opt_state = optimizer.init(variables)
loss, loss_and_grad = model.get_loss_ef_fn()
print('Optimizer initialized. Starting training...')

state_args = {'le_avg':0., 'lf_avg':0., 'loss_avg':0., 'iteration':0}
def train_step(batch, variables, opt_state, state_args, static_args):
    r = lr_scheduler(state_args['iteration']) / lr
    pref = {'e': s_pref_e*r + l_pref_e*(1-r), 'f': s_pref_f*r + l_pref_f*(1-r)}
    (loss_total, (loss_e, loss_f)), grads = loss_and_grad(variables, batch, pref, static_args)
    updates, opt_state = optimizer.update(grads, opt_state, variables)
    variables = optax.apply_updates(variables, updates)
    for key, value in zip(['loss_avg', 'le_avg', 'lf_avg'], [loss_total, loss_e, loss_f]):
        state_args[key] = state_args[key] * (1-1/l_smoothing) + value
    state_args['iteration'] += 1
    return variables, opt_state, state_args
train_step = jit(train_step, static_argnums=(4,))

def val_step(batch, variables, static_args):
    pref = {'e': 1, 'f': 1}
    _, (loss_e, loss_f) = loss(variables, batch, pref, static_args)
    return loss_e, loss_f
val_step = jit(val_step, static_argnums=(2,))

tic = time()
for iteration in range(total_steps):
    batch, _ = train_data.get_batch(batch_size)
    variables, opt_state, state_args = train_step(batch, variables, opt_state, state_args, static_args)
    if iteration % print_every == 0:
        if use_val_data:
            val_batch, _ = val_data.get_batch(val_batch_size)
            loss_val_e, loss_val_f = val_step(val_batch, variables, static_args)
        beta = l_smoothing * (1 - (1/l_smoothing)**(iteration+1))
        print('Iter %7d' % iteration +
              ' L %7.5f' % (state_args['loss_avg']/beta)**0.5 + 
              ' LE %7.5f' % ((state_args['le_avg']/beta)**0.5/train_data.natoms) +
              ' LF %7.5f' % (state_args['lf_avg']/beta)**0.5 + 
              (' LEval %7.5f' % ((loss_val_e)**0.5/val_data.natoms) if use_val_data else '') +
              (' LFval %7.5f' % (loss_val_f)**0.5 if use_val_data else '') +
              ' Time %.2fs' % (time()-tic))
        tic = time()

with open(save_name, 'wb') as file:
    pickle.dump({'model':model, 'variables':variables}, file)
T = int(time() - TIC)
print('Model saved to \'%s\'.' % save_name)
print('Training finished in %dh %dm %ds.' % (T//3600,(T%3600)//60,T%60))