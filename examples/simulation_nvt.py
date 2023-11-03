import os
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='90'
import jax.numpy as jnp
import numpy as np
from jax import jit, random, grad, vmap
import flax.linen as nn
import jax, sys, pickle, warnings
from time import time
from jax_md import space, quantity, simulate
sys.path.append(os.path.abspath('../'))
from deepmd_jax.data import SingleDataSystem
from deepmd_jax.dpmodel import DPModel
from deepmd_jax.utils import reorder_by_device
from deepmd_jax.simulation_utils import NeighborListLoader
K = jax.device_count() # Number of GPUs
shard_each = jax.sharding.PositionalSharding(jax.devices())
shard_all = shard_each.replicate()
print('Starting program on %d device(s):' % K, jax.devices())
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
np.set_printoptions(precision=4, suppress=True)
precision        = '32' # '16' '32' '64'
if precision == '32':
    jax.config.update('jax_default_matmul_precision', 'float32')
if precision == '64':
    jax.config.update('jax_enable_x64', True)

# units in Angstrom, eV, fs
model_path       = 'trained_models/model_water_mp_final_1.pkl'
save_prefix      = 'outdir/water_sim'
use_model_devi   = False    # compute model deviation of different models
model_devi_paths = ['model_water_new_2.pkl', 'model_water_new_3.pkl']
dt               = 0.48
temp             = 350 * 1.380649e-23 / 1.602176634e-19
mass             = np.concatenate([15.9994 * np.ones(128), 1.00784 * np.ones(256)]) * 1.66053907e-27 * 1e10 / 1.602176634e-19
use_neighborlist = True   # Do not use neighborlist if (1) box not orthorhombic or (2) max(rcut) over all models + max(rcut_buffer) larger than box/2
buffer_size      = 1.2    # Buffer for neighbor list (>1), increase it if there are frequent reallocations
update_every     = 8      # Frequency of neighbor list update, it'll be automatically lowered if rcut_buffer overflows
rcut_buffer      = [0.3, 0.7] # Within update_every steps, atoms(by type) should not move more than rcut_buffer/2
print_every      = 200    # Frequency of printing, as well as frequency of calculating model deviation
total_steps      = 1000000
dataset          = SingleDataSystem(['/pscratch/sd/r/ruiqig/polaron_cp2k/aimd/aimd-water/water_128/'], ['coord', 'box', 'force', 'energy'])
save_every       = 1       
chain_length     = 1           # Nose-Hoover chain length
tau              = 2000 * dt   # Nose-Hoover relaxation time
# prepare initial config numpy array in whatever way you like: coord (N,3), box (3,3) 
coord, box, force = dataset.data['coord'][0].T, dataset.data['box'][0].T.astype(jnp.float32), dataset.data['force'][0].T
L, M, N = 8, 8, 8
coord = np.concatenate([(coord + i*box[0])[:,None] for i in range(L)], axis=1).reshape(-1,3)
coord = np.concatenate([(coord + i*box[1])[:,None] for i in range(M)], axis=1).reshape(-1,3)
coord = np.concatenate([(coord + i*box[2])[:,None] for i in range(N)], axis=1).reshape(-1,3)
force = (force[:,None] * np.ones((L*M*N,1))).reshape(-1,3)
mass = (mass[:,None] * np.ones((L*M*N,))).reshape(-1)
box[0,0], box[1,1], box[2,2] = box[0,0]*L, box[1,1]*M, box[2,2]*N
type_idx = dataset.type_idx * (L*M*N)
# end
coord = jax.device_put(coord, shard_all)
box = jax.device_put(box, shard_all)
displace, shift = space.periodic(np.diag(box) if dataset.ortho else box)

with open(model_path, 'rb') as f:
    m = pickle.load(f)
model, variables = m['model'], jax.device_put(m['variables'], shard_all)
model.params['Ebias'] = model.params['Ebias'].astype(jnp.float32)
dataset.compute_lattice_candidate(model.params['rcut'])
model_list, variables_list = [model], [variables]
if use_model_devi:
    for path in model_devi_paths:
        with open(path, 'rb') as f:
            m = pickle.load(f)
            model_list.append(m['model']), variables_list.append(jax.device_put(m['variables'],shard_all))
nbrs_list = None
if use_neighborlist:
    rcut_all = max([m.params['rcut'] for m in model_list]) + np.array(rcut_buffer)
    rbuffer_array = jnp.concatenate([np.ones(type_idx[i+1]-type_idx[i],dtype=np.float32)*rcut_buffer[i] for i in range(len(rcut_buffer))])
    neighborlist = NeighborListLoader(np.diag(box), type_idx, rcut_all, buffer_size, K)
    coord = coord % np.diag(box)
    nbrs_list = neighborlist.allocate(coord.astype(jnp.float64))
    print('Neighborlist initialized with size', [nbrs.idx.shape[1] for nbrs in nbrs_list])
static_args = nn.FrozenDict({'lattice':dataset.get_batch(1)[1],
                             'type_idx':tuple(type_idx),
                             'ntype_idx':tuple(dataset.lattice_max*type_idx),
                             'K': K})
def get_energy_fn(model, variables):
    def energy_fn(coord, nbrs_list):
        coord = jax.device_put(reorder_by_device(coord, tuple(type_idx), K), shard_all)
        return model.apply(variables, coord.T, box.T, static_args, nbrs_list)[0]
    return jit(energy_fn)
energy_fn = get_energy_fn(model, variables)
energy_fns = [get_energy_fn(model, variables) for model, variables in zip(model_list, variables_list)]
@jit
def compute_model_devi(state, nbrs_list):
    all_forces = jnp.array([-grad(energy_fn)(state.position, nbrs_list) for energy_fn in energy_fns])
    return jnp.std(all_forces, axis=0).max()
print('Sanity check: NAtoms = ', len(coord), 'Energy = ', energy_fn(coord, nbrs_list),
      'Force error = ', ((force + jit(grad(energy_fn))(coord, nbrs_list))**2).mean()**0.5)
init_fn, apply_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, temp, chain_length=chain_length, tau=tau) 
state = init_fn(random.PRNGKey(0), coord, mass=mass, nbrs_list=nbrs_list)  
# init_fn, apply_fn = simulate.nve(energy, shift, dt)                                  # NVE
# state = init_fn(random.PRNGKey(0), coord, mass=mass, kT=temp, nbrs_list=nbrs_list)   # NVE
state_shard, nbrs_shard = jax.tree_util.tree_map(lambda x: x.sharding, state), jax.tree_util.tree_map(lambda x: x.sharding, nbrs_list)
def step_fn(states, i):
    state, nbrs_list = states
    state = apply_fn(state, nbrs_list=nbrs_list)
    return (state, nbrs_list), (state.position, state.velocity)
def get_multi_step_fn(steps):
    def multi_step_fn(states, i):
        state, nbrs_list = states
        (state_new, _), (pos, vel) = jax.lax.scan(step_fn, (state,nbrs_list), None, steps)
        nbrs_list = neighborlist.update(state.position, nbrs_list)
        rcut_overflow = (jnp.linalg.norm((state.position-pos-jnp.diag(box)/2)%jnp.diag(box)-jnp.diag(box)/2, axis=-1) > rbuffer_array/2).any()
        return (state_new, nbrs_list), (pos, vel, rcut_overflow)
    return multi_step_fn
multi_step_fn = get_multi_step_fn(update_every) if use_neighborlist else None
print('Step\tTemp\tKE\tPE\tInvariant\tModel Dev\ttime\tMomentum')
print('----------------------------------------')
pos_traj, vel_traj = [], []
i, tic = 0, time()
while i < total_steps:
    if i % print_every < (update_every if use_neighborlist else 1):
        PE = energy_fn(state.position, nbrs_list=nbrs_list)
        T = quantity.temperature(velocity=state.velocity, mass=state.mass) / 1.380649e-23 * 1.602176634e-19
        KE = quantity.kinetic_energy(velocity=state.velocity, mass=state.mass)
        H = simulate.nvt_nose_hoover_invariant(energy_fn, state, temp, nbrs_list=nbrs_list)
        model_devi = compute_model_devi(state, nbrs_list) if use_model_devi else 0.
        print('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.3f}\t{:.2f}'.format(
        i, T, KE, PE, H, model_devi, (time() - tic)), state.momentum.sum(axis=0))
        tic = time()
    if not use_neighborlist:
        (state, _), (pos, vel) = jax.lax.scan(step_fn, (state,None), None, print_every)
        i += print_every
    else:
        state, nbrs_list = jax.device_put(state, state_shard), jax.device_put(nbrs_list, nbrs_shard)
        (state_new, nbrs_list_new), (pos, vel, rcut_overflow) = jax.lax.scan(multi_step_fn, (state,nbrs_list), None, print_every//update_every)
        if rcut_overflow.any():
            if update_every == 1:
                print('Error: rcut_buffer overflow for a single step; Check for bugs or increase rcut_buffer')
            else:
                update_every = (update_every + 1) // 2
                multi_step_fn = get_multi_step_fn(update_every)
                print('Warning: rcut_buffer overflow; Decreased update_every to', update_every)
                continue
        if any([nbrs.did_buffer_overflow for nbrs in nbrs_list_new]):
            nbrs_list = neighborlist.allocate(state.position)
            nbrs_shard = jax.tree_util.tree_map(lambda x: x.sharding, nbrs_list)
            print('Warning: Neighbor list overflow; Reallocated with size', [nbrs.idx.shape[1] for nbrs in nbrs_list])
            continue
        state, nbrs_list = state_new, nbrs_list_new
        i += update_every * (print_every // update_every)
        pos, vel = pos.reshape(-1,pos.shape[2],3), vel.reshape(-1,vel.shape[2],3)
    pos_traj.append(np.array(pos[save_every-1::save_every]))
    vel_traj.append(np.array(pos[save_every-1::save_every]))
pos_traj, vel_traj = np.concatenate(pos_traj), np.concatenate(vel_traj)
# np.save(save_prefix + '_pos.npy', pos_traj)
# np.save(save_prefix + '_vel.npy', vel_traj)