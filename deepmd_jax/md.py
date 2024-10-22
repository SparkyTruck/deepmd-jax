import jax_md
import jax
import jax.numpy as jnp
import numpy as np
from utils import load_model
import flax.linen as nn
from time import time
from data import compute_lattice_candidate
from utils import split, concat

mass_unit_convertion = 1.036427e2 # from AMU to eV * fs^2 / Å^2
temperature_unit_convertion = 8.617333e-5 # from Kelvin to eV
pressure_unit_convertion = 6.241509e-7 # from bar to eV / Å^3

def get_energy_fn_from_potential(model, variables):
    def energy_fn(coord, cell, nbrs_nm, static_args):
        if True: # model type is energy
            E = model.apply(variables, coord, cell, static_args, nbrs_nm)[0]
        return E
    return jax.jit(energy_fn, static_argnames=('static_args'))

def get_static_args(position, rcut_maybe_with_buffer, type_count, box, use_neighbor_list=True):
    use_neighbor_list *= check_if_use_neighbor_list(box, rcut_maybe_with_buffer)
    if use_neighbor_list:
        static_args = nn.FrozenDict({'type_count':type_count, 'use_neighbor_list':True})
        return static_args
    else:
        lattice_args = compute_lattice_candidate(box[None], rcut_maybe_with_buffer)
        static_args = nn.FrozenDict({'type_count':type_count, 'lattice':lattice_args, 'use_neighbor_list':False})
        return static_args

def check_if_use_neighbor_list(box, rcut):
    if box.shape == (3,3):
        return False
    else:
        return False

class Simulation:
    _step_chunk_size: int = 10
    report_interval: int = 100
    step: int = 0
    dr_buffer_neighbor: float = 0.8
    dr_buffer_lattice: float = 1.
    neighbor_buffer_size: float = 1.2

    def __init__(self,
                 model_path,
                 type_idx,
                 box,
                 position,
                 mass,
                 routine,
                 dt,
                 velocity=None,
                 init_temperature=None,
                 **routine_args):
        if velocity is None and init_temperature is None:
            raise ValueError("Please either provide velocity or init_temperature to initialize velocity")
        self.energy_fn = get_energy_fn_from_potential(model_path, type_idx)
        self.dt = dt
        self.routine = routine
        self.routine_args = routine_args
        self.mass = jnp.array(np.array(mass)[np.array(type_idx)]) # AMU
        self.model, self.variables = load_model(model_path)
        self.energy_fn = get_energy_fn_from_potential(self.model, self.variables)
        type_count = np.bincount(type_idx.astype(int))
        self.type_count = np.pad(type_count, (0, self.model.params['ntypes'] - len(type_count)))

        # If orthorhombic, keep box as (3,); else keep (3,3)
        box = jnp.array(box)
        if box.size == 1:
            self.initial_box = box.item() * jnp.ones(3)
        if box.shape == (3,3) and (box == jnp.diag(jnp.diag(box))).all():
            self.initial_box = jnp.diag(box)
        self.current_box = self.initial_box

        # When box is variable, include extra buffer for lattice selection, and jax_md need fractional coordinates for shift_fn
        if "NPT" in self.routine:
            self.displacement_fn, self.shift_fn = jax_md.space.periodic_general(self.current_box)
            self.static_args = get_static_args(position,
                                               self.model.params['rcut'] + self.dr_buffer_lattice,
                                               self.type_count,
                                               self.current_box)
        else:
            self.displacement_fn, self.shift_fn = jax_md.space.periodic(self.current_box)
            self.static_args = get_static_args(position,
                                               self.model.params['rcut'],
                                               self.type_count,
                                               self.current_box)
        if self.static_args['use_neighbor_list']:
            self.nbrs = NeighborList(self.current_box,
                                       self.type_count,
                                       self.model.params['rcut'] + self.dr_buffer_neighbor,
                                       self.neighbor_buffer_size)
            self.nbrs.allocate(position, box=self.current_box)
        
        # Initialize according to routine;
        if self.routine == "NVE":
            self.routine_fn = jax_md.simulate.nve
        elif self.routine == "NVT_Nose_Hoover":
            self.routine_fn = jax_md.simulate.nvt_nose_hoover
            if 'temperature' not in routine_args:
                raise ValueError("Please provide extra argument 'temperature' for routine 'NVT_Nose_Hoover' in Kelvin")
            self.temperature = routine_args.pop('temperature')
            routine_args['kT'] = self.temperature * temperature_unit_convertion
        elif self.routine == "NPT_Nose_Hoover":
            self.routine_fn = jax_md.simulate.npt_nose_hoover
            if 'temperature' not in routine_args:
                raise ValueError("Please provide extra argument 'temperature' for routine 'NPT_Nose_Hoover' in Kelvin")
            if 'pressure' not in routine_args:
                raise ValueError("Please provide extra argument 'pressure' for routine 'NPT_Nose_Hoover' in bar")
            self.temperature = routine_args.pop('temperature')
            self.pressure = routine_args.pop('pressure')
            routine_args['kT'] = self.temperature * temperature_unit_convertion
            routine_args['pressure'] = self.pressure * pressure_unit_convertion
        else:
            raise NotImplementedError("routine is currently limited to 'NVE', 'NVT_Nose_Hoover', 'NPT_Nose_Hoover'")
        self.init_fn, self.apply_fn = self.routine_fn(self.energy_fn,
                                                     self.shift_fn,
                                                     dt,
                                                     **self.routine_args)
        self.state = self.init_fn(jax.random.PRNGKey(0),
                                  position,
                                  mass=self.mass * mass_unit_convertion,
                                  kT=((init_temperature if init_temperature is not None else 0)
                                      * temperature_unit_convertion),
                                  nbrs=self.nbrs.nbrs_nm if self.static_args['use_neighbor_list'] else None,
                                  static_args=self.static_args,
                                  cell=self.current_box,
                                  **({'box':self.current_box} if "NPT" in self.routine else {}))
        if init_temperature is None:
            self.state.set(velocity=velocity)

    def generate_report_fn(self):
        energy_fn_kwargs = {"cell":self.state.box if "NPT" in self.routine else self.initial_box,
                            "nbrs_nm":self.nbrs.nbrs_nm if self.static_args['use_neighbor_list'] else None,
                            "static_args":self.static_args}
        self.reporters = {
            "Temperature": lambda state: jax_md.quantity.temperature(
                                                            velocity=state.velocity,
                                                            mass=state.mass,
                                                        ) / temperature_unit_convertion,
            "KE": lambda state: jax_md.quantity.kinetic_energy(
                                                    velocity=state.velocity,
                                                    mass=state.mass,
                                                ),
            "PE": lambda state: self.energy_fn(
                                    state.position,
                                    **energy_fn_kwargs,
                                ),                 
            }

        if self.routine == "NVT_Nose_Hoover":
            self.reporters["Invariant"] = lambda state: jax_md.simulate.nvt_nose_hoover_invariant(
                                            self.energy_fn,
                                            state,
                                            self.temperature,
                                            **energy_fn_kwargs
                                        )
        elif self.routine == "NPT_Nose_Hoover":
            self.reporters["Invariant"] = lambda state: jax_md.simulate.npt_nose_hoover_invariant(
                                            self.energy_fn,
                                            state,
                                            self.temperature,
                                            self.pressure,
                                            **energy_fn_kwargs
                                        )
        def report_fn(state):
            return [fn(state) for fn in self.reporters.values()]
        return jax.jit(report_fn)
    
    def make_report(self, initial=False):
        char_space = ["8"] + ["12"] * len(self.reporters) + ["6"]
        if initial:
            self.tic = time()
            headers = ["Step"] + list(self.reporters.keys()) + ["Time"]
            print("|".join([f"{header:^{char_space[i]}}" for i, header in enumerate(headers)]))
        report = [self.step] + self.report_fn(self.state) + [time()-self.tic]
        print("|".join([f"{value:^{char_space[i]}.3f}" for i, value in enumerate(report)]))
        self.tic = time()
            
    def check_lattice_overflow(self, position, box):
        '''Overflow that requires increasing lattice candidate/buffer, not jit-compatible'''
        pass
        return False

    def check_hard_overflow(self, box):
        '''Overflow that requires disabling neighbor list, not jit-compatible'''
        if box == None: # Not variable-box
            return False
        else:
            pass
            return False
    
    def check_soft_overflow(self, position, ref_position, box):
        '''Movement over dr_buffer_neighbor/2 that requires neighbor update, jit-compatible'''
        return False

    def get_inner_step(self):
        def inner_step(states):
            state, nbrs, overflow = states
            npt_box = state.box if "NPT" in self.routine else None
            current_box = state.box if "NPT" in self.routine else self.current_box
            soft_overflow = self.check_soft_overflow(state.position, nbrs.reference_position, current_box)
            nbrs = jax.lax.cond(soft_overflow,
                                lambda nbrs: nbrs.update(state.position, box=npt_box),
                                lambda nbrs: nbrs,
                                nbrs)
            state = self.apply_fn(state,
                                  cell=current_box,
                                  nbrs=nbrs.nbrs_nm if self.static_args['use_neighbor_list'] else None,
                                  static_args=self.static_args)
            is_nbr_buffer_overflow, is_lattice_overflow, is_hard_overflow = overflow
            if self.static_args['use_neighbor_list']:
                is_nbr_buffer_overflow |= nbrs.did_buffer_overflow
                is_hard_overflow |= self.check_hard_overflow(npt_box)
            else:
                is_lattice_overflow |= self.check_lattice_overflow(state.position, current_box)
            overflow = (is_nbr_buffer_overflow, is_lattice_overflow, is_hard_overflow)
            return ((state, nbrs, overflow),
                    (state.position, state.velocity))
        return inner_step

    def run(self, steps):

        position_trajectory, velocity_trajectory = [], []
        self.inner_step_fn = self.get_inner_step()

        # Make an initial report
        self.report_fn = self.generate_report_fn()
        self.make_report(initial=True)

        # steps stands for the remaining steps to run
        while steps > 0:

            # run the simulation for a chunk of steps in a lax.scan loop
            next_chunk = min(self.report_interval - self.step % self.report_interval,
                        self._step_chunk_size - self.step % self._step_chunk_size)
            states = (self.state,
                      self.nbrs if self.static_args['use_neighbor_list'] else None,
                      (False,) * 3)
            states_new, (pos_traj, vel_traj) = jax.lax.scan(self.inner_step_fn, states, None, next_chunk)
            state_new, nbrs_new, overflow = states_new
            is_nbr_buffer_overflow, is_lattice_overflow, is_hard_overflow = overflow

            # If anything overflows, we have to re-run the chunk
            if is_hard_overflow or is_lattice_overflow or is_nbr_buffer_overflow:
                if is_hard_overflow: # Need to disable neighbor list
                    self.static_args = get_static_args(self.state.position,
                                                    self.model.params['rcut'],
                                                    self.type_count,
                                                    state_new.box,
                                                    use_neighbor_list=False)
                elif is_nbr_buffer_overflow: # Need to reallocate neighbor list
                    self.nbrs.allocate(self.state.position, box=self.current_box)
                elif is_lattice_overflow: # Need to increase lattice candidate/neighbor count
                    self.static_args = get_static_args(self.state.position,
                                                    self.model.params['rcut'] + self.dr_buffer_lattice,
                                                    self.type_count,
                                                    state_new.box)
                self.inner_step_fn = self.get_inner_step()
                self.report_fn = self.generate_report_fn()
                continue

            # If nothing overflows, update the state and record the trajectory
            self.state = state_new
            self.nbrs = nbrs_new
            self.step += next_chunk
            steps -= next_chunk
            position_trajectory.append(np.array(pos_traj))
            velocity_trajectory.append(np.array(vel_traj))

            # Report at preset regular intervals
            if self.step % self.report_interval == 0:
                self.make_report()
        
        # Return the trajectory
        position_trajectory = np.concatenate(position_trajectory)
        velocity_trajectory = np.concatenate(velocity_trajectory)
        return position_trajectory, velocity_trajectory

def get_mask_by_device(type_count):
    '''
        For multiple-device partitioning, ghost atoms are padded.
        Returns a binary mask indicating the valid atoms, sharded by device.
    '''
    K = jax.device_count()
    mask = concat([
                concat([jnp.ones(count, dtype=bool),
                        jnp.zeros((-count%K,), dtype=bool)
                        ]).reshape(K,-1)
                for count in type_count
                ],
            axis=1).reshape(-1)
    # ensure mask is sharded by device
    sharding = jax.sharding.PositionalSharding(jax.devices())
    return jax.lax.with_sharding_constraint(mask, sharding)

def get_type_mask_fns(type_count):
    '''
        Returned type_mask_fns filter nbrs.idx by atom type.
        Padded ghost atoms are excluded.
        This can be used to postprocess neighbor indices,
        and can also be added as a custom_mask_function in jax_md.neighbor_list.
    '''
    type_mask_fns = []
    K = jax.device_count()
    full_mask = get_mask_by_device(type_count)
    type_count_new = -(-type_count//K) # type_count for each device after padding
    type_idx_filled_each = np.cumsum(np.concatenate([[0], type_count_new]))
    N_each = type_idx_filled_each[-1] # number of atoms for each device after padding
    for i in range(len(type_count)):
        # filter neighbor atoms by type i; idx = nbrs.idx returned by jax_md
        def mask_fn(idx, i=i): 
            # ensure idx is sharded by device
            sharding = jax.sharding.PositionalSharding(jax.devices()).reshape(K,1)
            idx = jax.device_put(idx, sharding)
            valid_center_atom = full_mask[:,None]
            valid_type_neighbor_atom = idx%N_each >= type_idx_filled_each[i]
            valid_type_neighbor_atom *= idx%N_each < type_idx_filled_each[i+1]
            valid_neighbor_atom = (idx-type_idx_filled_each[i]-(idx//N_each)*(N_each-type_count_new[i]) < type_count[i])
            is_neighbor = (idx < N_each * K)
            filter = valid_center_atom * valid_type_neighbor_atom * valid_neighbor_atom * is_neighbor
            return jnp.where(filter, idx, N_each * K)
        type_mask_fns.append(mask_fn)
    return type_mask_fns

def get_idx_mask_fn(type_count):
    '''
        Returned idx_mask_fn that filters out ghost atoms from nbrs.idx.
    '''
    full_mask = get_mask_by_device(type_count)
    idx_mask_out = np.arange(len(full_mask))[~np.array(full_mask)]
    def idx_mask_fn(idx):
        # ensure idx is sharded by device
        sharding = jax.sharding.PositionalSharding(jax.devices()).reshape(jax.device_count(),1)
        idx = jax.device_put(idx, sharding)
        filter = full_mask[:,None] * jnp.isin(idx, idx_mask_out, invert=True)
        return jnp.where(filter, idx, len(full_mask))
    return idx_mask_fn

from typing import Callable
@jax_md.dataclasses.dataclass
class TypedNeighborList():
    nbrs: jax_md.partition.NeighborList
    type_count: tuple
    idx_mask_fn: list[Callable]
    type_mask_fns: list[Callable]
    rcut: float
    @property
    def did_buffer_overflow(self) -> bool:
        return self.nbrs.did_buffer_overflow
    @property
    def idx(self) -> jax.Array:
        return self.nbrs.idx
    @property
    def reference_position(self) -> jax.Array:
        return self.nbrs.reference_position

def type_neighbor_list_fn():

    def __init__(self, box, type_count, rcut, size):
        self.type_count, self.box = tuple(type_count), box.astype(jnp.float32)
        self.mask_fns = get_type_mask_fns(np.array(type_count))
        self.mask_fn = get_full_mask_fn(np.array(type_count))
        self.rcut, self.size = rcut, size
    def canonicalize(self, coord):
        coord = (coord.astype(jnp.float32) % self.box) * (1-2e-7) + 1e-7*self.box # avoid numerical error at box boundary
        return reorder_by_device(coord, self.type_count)
    def allocate(self, coord):
        displace = space.periodic(self.box)[0]
        coord = self.canonicalize(coord)
        test_nbr = partition.neighbor_list(displace, self.box, self.rcut, capacity_multiplier=1.,
                                           custom_mask_function=self.mask_fn).allocate(coord)
        self.knbr = np.array([int(((fn(test_nbr.idx)<len(coord)).sum(1).max())*self.size) for fn in self.mask_fns])
        self.knbr = np.where(self.knbr==0, 1, self.knbr + 1 + max(int(20*(self.size-1.2)),0))
        buffer = (sum(self.knbr)+1) / test_nbr.idx.shape[1]
        print('# Neighborlist allocated with size', np.array(self.knbr)-1)
        return partition.neighbor_list(displace, self.box, self.rcut, capacity_multiplier=buffer,
                                        custom_mask_function=self.mask_fn).allocate(coord)
    def update(self, coord, nbrs):
        return nbrs.update(self.canonicalize(coord))
    def check_dr_overflow(self, coord, ref, dr_buffer):
        return (jnp.linalg.norm((coord-ref-self.box/2)
                    %self.box - self.box/2, axis=-1) > dr_buffer/2 - 0.01).any()
    def get_nm(self, nbrs):
        K = jax.device_count()
        sharding = jax.sharding.PositionalSharding(jax.devices()).reshape(K, 1)
        nbr_idx = lax.with_sharding_constraint(nbrs.idx, sharding)
        nbrs_idx = [-lax.top_k(-fn(nbr_idx), self.knbr[i])[0] for i, fn in enumerate(self.mask_fns)]
        type_count_new = [-(-self.type_count[i]//K) for i in range(len(self.type_count))]
        type_idx_new = np.cumsum([0] + list(type_count_new))
        nbrs_nm = [mlist for mlist in zip(*[split(jnp.where(nbrs < type_idx_new[-1]*K,
            nbrs - type_idx_new[i] - (nbrs//type_idx_new[-1]) * (type_idx_new[-1]-type_count_new[i]),
            type_idx_new[-1]*K), type_count_new, K=K) for i, nbrs in enumerate(nbrs_idx)])]
        overflow = jnp.array([(idx.max(axis=1)<type_idx_new[-1]*K).any() for idx in nbrs_idx]).any() | nbrs.did_buffer_overflow
        return nbrs_nm, overflow