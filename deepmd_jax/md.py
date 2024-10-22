import jax_md
import jax
import jax.numpy as jnp
import numpy as np
from utils import load_model
import flax.linen as nn
from time import time
from data import compute_lattice_candidate
from utils import split, concat
from typing import Callable

mass_unit_convertion = 1.036427e2 # from AMU to eV * fs^2 / Å^2
temperature_unit_convertion = 8.617333e-5 # from Kelvin to eV
pressure_unit_convertion = 6.241509e-7 # from bar to eV / Å^3

def get_energy_fn_from_potential(model, variables):
    '''
        Returns a energy function from a deepmd model.
        You can define a custom energy function with the same signature,
        and setting the energy_fn property of a Simulation instance.
    '''
    def energy_fn(coord, cell, nbrs_nm, static_args):
        if True: # model type is energy
            E = model.apply(variables, coord, cell, static_args, nbrs_nm)[0]
        return E
    return jax.jit(energy_fn, static_argnames=('static_args'))

def get_static_args(position, rcut_maybe_with_buffer, type_count, box, use_neighbor_list=True):
    '''
        Returns a FrozenDict of the complete set of static arguments for jit compilation.
    '''
    use_neighbor_list *= check_if_use_neighbor_list(box, rcut_maybe_with_buffer)
    if use_neighbor_list:
        static_args = nn.FrozenDict({'type_count':type_count, 'use_neighbor_list':True})
        return static_args
    else:
        lattice_args = compute_lattice_candidate(box[None], rcut_maybe_with_buffer)
        static_args = nn.FrozenDict({'type_count':type_count, 'lattice':lattice_args, 'use_neighbor_list':False})
        return static_args

def check_if_use_neighbor_list(box, rcut):
    '''
        Neighbor list currently only allowed for a sufficiently big orthorhombic box.
    '''
    if box.shape == (3,3):
        return False
    else:
        return False

class Simulation:
    '''
        A deepmd-based simulation class that wraps jax_md.simulate routines.
        Targeting a fully automatic use: initialize and run(steps).
    '''

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
            self.displacement_fn, self.shift_fn = jax_md.space.periodic_general(self.initial_box)
            self.static_args = get_static_args(position,
                                               self.model.params['rcut'] + self.dr_buffer_lattice,
                                               self.type_count,
                                               self.initial_box)
        else:
            self.displacement_fn, self.shift_fn = jax_md.space.periodic(self.initial_box)
            self.static_args = get_static_args(position,
                                               self.model.params['rcut'],
                                               self.type_count,
                                               self.initial_box)
        # Initialize neighbor list if needed
        if self.static_args['use_neighbor_list']:
            self.construct_nbr_and_nbr_fn(self.dr_buffer_neighbor, self.neighbor_buffer_size)
        
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
                                  nbrs=self.typed_nbrs.nbrs_nm if self.static_args['use_neighbor_list'] else None,
                                  static_args=self.static_args,
                                  cell=self.initial_box,
                                  **({'box':self.initial_box} if "NPT" in self.routine else {}))
        if init_temperature is None:
            self.state.set(velocity=velocity)

    def generate_report_fn(self):
        energy_fn_kwargs = {"cell":self.state.box if "NPT" in self.routine else self.initial_box,
                            "nbrs_nm":self.typed_nbrs.nbrs_nm if self.static_args['use_neighbor_list'] else None,
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

    def construct_nbr_and_nbr_fn(self, dr_buffer_neighbor, neighbor_buffer_size):
        '''
            Initial construction, or reconstruction when allocate cannot handle the overflow.
        '''
        self.typed_nbr_fn = typed_neighbor_list(self.current_box,
                                                self.type_count,
                                                self.model.params['rcut'] + dr_buffer_neighbor,
                                                neighbor_buffer_size)
        self.typed_nbrs = self.typed_nbr_fn.allocate(self.state.position, box=self.current_box)
            
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
            state, typed_nbrs, overflow = states
            npt_box = state.box if "NPT" in self.routine else None
            current_box = state.box if "NPT" in self.routine else self.initial_box
            soft_overflow = self.check_soft_overflow(state.position, typed_nbrs.reference_position, current_box)
            typed_nbrs = jax.lax.cond(soft_overflow,
                                lambda typed_nbrs: self.typed_nbr_fn.update(state.position, box=npt_box),
                                lambda typed_nbrs: typed_nbrs,
                                typed_nbrs)
            state = self.apply_fn(state,
                                  cell=current_box,
                                  nbrs=typed_nbrs.nbrs_nm if self.static_args['use_neighbor_list'] else None,
                                  static_args=self.static_args)
            is_nbr_buffer_overflow, is_lattice_overflow, is_hard_overflow = overflow
            if self.static_args['use_neighbor_list']:
                is_nbr_buffer_overflow |= typed_nbrs.did_buffer_overflow
                is_hard_overflow |= self.check_hard_overflow(npt_box)
            else:
                is_lattice_overflow |= self.check_lattice_overflow(state.position, current_box)
            overflow = (is_nbr_buffer_overflow, is_lattice_overflow, is_hard_overflow)
            return ((state, typed_nbrs, overflow),
                    (state.position,
                     state.velocity, 
                     state.box if "NPT" in self.routine else self.initial_box,
                    ))
        return inner_step

    def run(self, steps):
        '''
            Usage: trajectory = run(steps)
        '''
        position_trajectory, velocity_trajectory, box_trajectory = [], [], []
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
                      self.typed_nbrs if self.static_args['use_neighbor_list'] else None,
                      (False,) * 3)
            states_new, (pos_traj, vel_traj, box_traj) = jax.lax.scan(self.inner_step_fn,
                                                                      states,
                                                                      length=next_chunk)
            state_new, typed_nbrs_new, overflow = states_new
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
                    self.typed_nbrs = self.typed_nbr_fn.allocate(self.state.position,
                                                                 box=self.state.box if "NPT" in self.routine else None)
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
            self.typed_nbrs = typed_nbrs_new
            if "NPT" in self.routine:
                self.current_box = self.state.box
            self.step += next_chunk
            steps -= next_chunk
            position_trajectory.append(np.array(pos_traj))
            velocity_trajectory.append(np.array(vel_traj))
            box_trajectory.append(np.array(box_traj))

            # Report at preset regular intervals
            if self.step % self.report_interval == 0:
                self.make_report()
        
        # Return the trajectory
        trajectory = {
            'position': np.concatenate(position_trajectory),
            'velocity': np.concatenate(velocity_trajectory),
            'box': np.concatenate(box_trajectory)
            }
        return trajectory
    
def reorder_by_device(coord, type_count):
    '''
        For multiple devices, ghost atoms are padded to ensure equal partitioning.
        Each type of atom is padded and partitioned separately and then concatenated.
    '''
    K = jax.device_count()
    coord = jnp.concatenate(
                [
                    jnp.pad(c,
                            ((0,-c.shape[0]%K),)+((0,0),)*(c.ndim-1)
                        ).reshape(K,-1,*c.shape[1:])
                    for c in split(coord,type_count)
                ], axis=1).reshape(-1, *coord.shape[1:])
    sharding = jax.sharding.PositionalSharding(jax.devices())
    return jax.lax.with_sharding_constraint(coord, sharding.replicate())

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
    type_count_each = -(-np.array(type_count)//K)        # type_count for each device after padding
    type_idx_each = np.cumsum([0] + list(type_count_each)) # type_idx for each device after padding
    N_each = type_idx_each[-1] # number of atoms for each device after padding
    for i in range(len(type_count)):
        # filter neighbor atoms by type i; idx = nbrs.idx returned by jax_md
        def mask_fn(idx, i=i): 
            # ensure idx is sharded by device
            sharding = jax.sharding.PositionalSharding(jax.devices()).reshape(K,1)
            idx = jax.device_put(idx, sharding)
            valid_center_atom = full_mask[:,None]
            valid_type_neighbor_atom = idx%N_each >= type_idx_each[i]
            valid_type_neighbor_atom *= idx%N_each < type_idx_each[i+1]
            valid_neighbor_atom = (idx-type_idx_each[i]-(idx//N_each)*(N_each-type_count_each[i]) < type_count[i])
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

@jax_md.dataclasses.dataclass
class TypedNeighborList():
    '''
        Wraps jax_md.partition.NeighborList with typed lists.
    '''
    nbrs: jax_md.partition.NeighborList
    knbr: list[int]
    nbrs_nm: list[list[jnp.ndarray]]
    did_buffer_overflow: bool
    @property
    def idx(self) -> jax.Array:
        return self.nbrs.idx
    @property
    def reference_position(self) -> jax.Array:
        return self.nbrs.reference_position
    
@jax_md.dataclasses.dataclass
class typed_neighbor_list_fn():
    '''A struct containing functions to allocate and update neighbor lists.'''
    allocate: Callable = jax_md.dataclasses.field()
    update: Callable = jax_md.dataclasses.field()

def typed_neighbor_list(box, type_count, rcut, buffer_size, fractional=False):
    type_count = tuple(type_count)
    box = box.astype(jnp.float32)
    type_mask_fns = get_type_mask_fns(type_count)
    idx_mask_fn = get_idx_mask_fn(type_count)

    # store coord in float32, reorder by device, and shrink by 1e-7 to avoid numerical issues
    def canonicalize(coord):
        coord = (coord.astype(jnp.float32) % box) * (1-2e-7) + 1e-7 * box
        return reorder_by_device(coord, type_count)
    
    # allocate function: non-jit-compatible
    def allocate_fn(coord):
        displacement_fn = jax_md.space.periodic(box)[0]
        coord = canonicalize(coord)
        # Measure the exact size of max neighbors for each atom type
        test_nbr = jax_md.partition.neighbor_list(displacement_fn,
                                                  box,
                                                  rcut,
                                                  capacity_multiplier=1.,
                                                  custom_mask_function=idx_mask_fn
                                                ).allocate(coord)
        knbr = np.array([
                (type_mask(test_nbr.idx) < len(coord)).sum(1).max()
                for type_mask in type_mask_fns
            ])
        # Bloat knbr by buffer_size; also ensure if buffer_size+=0.05, knbr at least +1
        knbr = np.where(knbr == 0,
                        1,
                        1 + int(knbr * buffer_size + 
                            max(20 - knbr,0) * max(buffer_size-1.2,0)))
        print('# Neighborlist allocated with size', np.array(knbr) - 1)
        # infer a total buffer from the max neighbors of each type
        buffer = (sum(knbr)+1.01) / test_nbr.idx.shape[1]
        # Allocate the neighbor list with the inferred buffer ratio
        nbrs = jax_md.partition.neighbor_list(displacement_fn,
                                              box,
                                              rcut,
                                              capacity_multiplier=buffer,
                                              custom_mask_function=idx_mask_fn
                                            ).allocate(coord)
        nbrs_nm, overflow = get_nm(nbrs, knbr)
        return TypedNeighborList(nbrs=nbrs,
                                 knbr=knbr,
                                 nbrs_nm=nbrs_nm,
                                 did_buffer_overflow=overflow)
    
    # update function: jit-compatible
    def update_fn(coord, typed_nbrs):
        typed_nbrs = typed_nbrs.set(nbrs=typed_nbrs.nbrs.update(canonicalize(coord)))
        nbrs_nm, overflow = get_nm(typed_nbrs.nbrs, typed_nbrs.knbr)
        return typed_nbrs.set(nbrs_nm=nbrs_nm,
                              did_buffer_overflow=overflow|typed_nbrs.did_buffer_overflow)
    
    # def check_dr_overflow(coord, reference_coord, dr_buffer, box):
    #     return (jnp.linalg.norm(
    #                 (coord- reference_coord - box/2) % box - box/2,
    #                 axis=-1
    #             ) > dr_buffer/2 - 0.01).any()
    
    def get_nm(nbrs, knbr):
        '''
            Get neighbor idx as a nested list of device-sharded arrays.
            nbrs_nm[i][j] for type i center atoms with neighbor type j.
            Entries stand for atom indices after device partitioning.
        '''

        # ensure idx is sharded by device
        K = jax.device_count()
        sharding = jax.sharding.PositionalSharding(jax.devices()).reshape(K, 1)
        nbr_idx = jax.lax.with_sharding_constraint(nbrs.idx, sharding)

        # take knbr[i] smallest indices for type i
        nbrs_idx = [-jax.lax.top_k(-fn(nbr_idx), knbr[i])[0]
                    for i, fn in enumerate(type_mask_fns)]
        type_count_each = -(-np.array(type_count)//K)        # type_count for each device after padding
        type_idx_each = np.cumsum([0] + list(type_count_each)) # type_idx for each device after padding
        # convert idx to each type pair
        nbrs_nm = [mlist for mlist in 
                   zip(*[split(jnp.where(nbrs < type_idx_each[-1]*K,
                                         nbrs - type_idx_each[i] - (nbrs//type_idx_each[-1]) * (type_idx_each[-1] - type_count_each[i]),
                                         type_idx_each[-1]*K
                                    ), type_count_each, K=K)
                        for i, nbrs in enumerate(nbrs_idx)
                        ])]
        overflow = nbrs.did_buffer_overflow | jnp.array([
                                    (idx.max(axis=1)<type_idx_each[-1]*K).any()
                                    for idx in nbrs_idx
                                    ]).any()
        return nbrs_nm, overflow
    
    return typed_neighbor_list_fn(allocate=allocate_fn, update=update_fn)