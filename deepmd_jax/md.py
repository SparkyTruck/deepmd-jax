import jax_md
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from time import time
from .data import compute_lattice_candidate
from .utils import split, concat, load_model, norm_ortho_box
from typing import Callable

MASS_UNIT_CONVERSION = 1.036427e2 # from AMU to eV * fs^2 / Å^2
TEMP_UNIT_CONVERSION = 8.617333e-5 # from Kelvin to eV
PRESS_UNIT_CONVERSION = 6.241509e-7 # from bar to eV / Å^3

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
            idx = jax.lax.with_sharding_constraint(idx, sharding)
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
        idx = jax.lax.with_sharding_constraint(idx, sharding)
        filter = full_mask[:,None] * jnp.isin(idx, idx_mask_out, invert=True)
        return jnp.where(filter, idx, len(full_mask))
    return idx_mask_fn

@jax_md.dataclasses.dataclass
class TypedNeighborList():
    '''
        Wraps jax_md.partition.NeighborList with typed lists.
    '''
    nbrs: jax_md.partition.NeighborList
    knbr: list[int] = jax_md.dataclasses.static_field()
    nbrs_nm: list[list[jnp.ndarray]]
    reference_box: jax.Array
    did_buffer_overflow: bool = False
    @property
    def idx(self) -> jax.Array:
        return self.nbrs.idx
    @property
    def reference_position(self) -> jax.Array:
        return self.nbrs.reference_position
    
@jax_md.dataclasses.dataclass
class typed_neighbor_list_fn():
    '''A struct containing functions to allocate and update neighbor lists.'''
    allocate: Callable = jax_md.dataclasses.static_field()
    update: Callable = jax_md.dataclasses.static_field()

def typed_neighbor_list(box, type_idx, rcut, buffer_ratio=1.2):
    '''
        Returns a typed neighbor list function that can be used to allocate and update neighbor lists.
    '''
    type_idx = np.array(type_idx, dtype=int)
    type_count = tuple(np.bincount(type_idx))
    reference_box = box.astype(jnp.float32) # box at creation of typed_neighbor_list_fn; only pass this box to jax_md.neighbor_list
    type_mask_fns = get_type_mask_fns(type_count)
    idx_mask_fn = get_idx_mask_fn(type_count)
    
    def canonicalize(coord, box=reference_box):
        # scale coord to reference_box and store in float32 fractional coordinates
        box = box.astype(jnp.float32)
        scale = reference_box[0] / box[0]
        coord = coord[type_idx.argsort(kind='stable')]
        coord = (coord.astype(jnp.float32) % box) * scale
        # shrink by 1e-7 from the boundary to avoid numerical issues
        coord = coord * (1-2e-7) + 1e-7 * reference_box
        # reorder and partition to multiple devices
        coord = reorder_by_device(coord, type_count)
        return coord
    
    # allocate function: non-jit-compatible
    def allocate_fn(coord, box=reference_box):
        coord = canonicalize(coord, box)
        displacement_fn = jax_md.space.periodic(reference_box)[0]
        # Measure the exact size of max neighbors for each atom type
        test_nbr = jax_md.partition.neighbor_list(displacement_fn,
                                                  reference_box,
                                                  rcut * reference_box[0] / box[0], # scale rcut to reference_box
                                                  capacity_multiplier=1.,
                                                  custom_mask_function=idx_mask_fn
                                                ).allocate(coord)
        knbr = np.array([
                (type_mask(test_nbr.idx) < len(coord)).sum(1).max()
                for type_mask in type_mask_fns
            ])
        # Bloat knbr by buffer_ratio; also ensure if buffer_ratio += 0.05, knbr at least +1
        knbr = np.where(knbr == 0,
                        1,
                        1 + (knbr * buffer_ratio + 
                            np.maximum(20 - knbr,0) * max(buffer_ratio-1.2,0)))
        knbr = list(knbr.astype(int))
        print(f'# Neighborlist allocated with size {np.array(knbr) - 1}, rcut = {rcut}, buffer_ratio = {buffer_ratio}')
        # infer a total buffer from the max neighbors of each type
        total_buffer_ratio = (sum(knbr)+1.01) / test_nbr.idx.shape[1]
        # Allocate the neighbor list with the inferred buffer ratio
        nbrs = jax_md.partition.neighbor_list(displacement_fn,
                                              box,
                                              rcut * reference_box[0] / box[0], # scale rcut to reference_box
                                              capacity_multiplier=total_buffer_ratio,
                                              custom_mask_function=idx_mask_fn
                                            ).allocate(coord)
        nbrs_nm, overflow = get_nm(nbrs, knbr)
        return TypedNeighborList(nbrs=nbrs,
                                 knbr=knbr,
                                 nbrs_nm=nbrs_nm,
                                 did_buffer_overflow=overflow,
                                 reference_box=reference_box)
    
    # update function: jit-compatible
    def update_fn(coord, typed_nbrs, box=reference_box):
        coord = canonicalize(coord, box)
        typed_nbrs = typed_nbrs.set(nbrs=typed_nbrs.nbrs.update(coord))
        nbrs_nm, overflow = get_nm(typed_nbrs.nbrs, typed_nbrs.knbr)
        return typed_nbrs.set(nbrs_nm=nbrs_nm,
                              did_buffer_overflow=overflow|typed_nbrs.did_buffer_overflow)
    
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
        nbrs_idx = [-jax.lax.top_k(-fn(nbr_idx), k)[0]
                    for fn,k in zip(type_mask_fns, knbr)]
        type_count_each = -(-np.array(type_count)//K)        # type_count for each device after padding
        type_idx_each = np.cumsum([0] + list(type_count_each)) # type_idx for each device after padding
        # convert idx to each type pair; this is a bit tricky
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

class Simulation:
    '''
        A deepmd-based simulation class that wraps jax_md.simulate routines.
        Targeting a fully automatic use: initialize and run(steps).
    '''

    step: int = 0
    report_interval: int = 100
    _step_chunk_size: int = 10
    _dr_buffer_neighbor: float = 0.8
    _dr_buffer_lattice: float = 1.
    _neighbor_buffer_ratio: float = 1.2
    _neighbor_update_profile: jax.Array = jnp.array([0,0])
    _error_code: int = 0
    _is_initial_state: bool = True

    def __init__(self,
                 model_path,
                 box,
                 type_idx,
                 mass,
                 routine,
                 dt,
                 initial_position,
                 initial_velocity=None,
                 temperature=None,
                 debug=False,
                 model_deviation_paths=[],
                 **routine_args):
        '''
            Initialize a Simulation instance.
            model_path: path to the deepmd model
            box: box size, scalar or (3,) or (3,3)
            type_idx: list of atom types, length = n_atoms
            mass: atomic mass for each type, length = number of atom types
            routine: simulation routine, currently limited to 'NVE', 'NVT_Nose_Hoover', 'NPT_Nose_Hoover'
            dt: time step size (fs)
            initial_position: shape = (n_atoms, 3)
            initial_velocity: shape = (n_atoms, 3); if None, use temperature to generate; if temperature is also None, use 0.
            temperature: temperature in NVT/NPT simulations, or NVE initial_velocity generation if initial_velocity is None.
            debug: print additional debug information
            model_deviation_paths: a list of deepmd models for deviation calculation used in active learning
            routine_args: extra arguments for the simulation routine, check the jax_md.simulate routines for details.

            Usage:
                sim = Simulation(...)
                trajectory = sim.run(steps)
        '''
        initial_position = jnp.array(initial_position) * jnp.ones(1) # Ensure default precision
        self._dt = dt
        self._routine = routine
        self._routine_args = routine_args
        self._temperature = temperature
        self._type_idx = np.array(type_idx.astype(int))
        self._mass = jnp.array(np.array(mass)[np.array(self._type_idx)]) # AMU
        self._model, self._variables = load_model(model_path)
        type_count = np.bincount(self._type_idx)
        self._type_count = np.pad(type_count, (0, self._model.params['ntypes'] - len(type_count)))
        self._debug = debug
        self._model_deviation_paths = model_deviation_paths
        self._use_model_deviation = len(model_deviation_paths) > 0
        # If orthorhombic, keep box as (3,); else keep (3,3)
        box = jnp.array(box)
        if box.size == 1:
            self._initial_box = box.item() * jnp.ones(3)
        if box.shape == (3,3) and (box == jnp.diag(jnp.diag(box))).all():
            self._initial_box = jnp.diag(box)
        self._current_box = self._initial_box
        self._static_args = self._get_static_args(initial_position)
        self._displacement_fn, self._shift_fn = jax_md.space.periodic_general(self._initial_box,
                                                                              fractional_coordinates="NPT" in self._routine)
        # Initialize neighbor list if needed
        if self._static_args['use_neighbor_list']:
            self._construct_nbr_and_nbr_fn(initial_position)
        
        # Initialize according to routine;
        if self._routine == "NVE":
            self._routine_fn = jax_md.simulate.nve
        elif self._routine == "NVT_Nose_Hoover":
            self._routine_fn = jax_md.simulate.nvt_nose_hoover
            self._routine_args['kT'] = self._temperature * TEMP_UNIT_CONVERSION
        elif self._routine == "NPT_Nose_Hoover":
            self._routine_fn = jax_md.simulate.npt_nose_hoover
            if 'pressure' not in routine_args:
                raise ValueError("Please provide extra argument 'pressure' (in bar) for routine 'NPT_Nose_Hoover'")
            self._pressure = routine_args['pressure']
            self._routine_args['kT'] = self._temperature * TEMP_UNIT_CONVERSION
            self._routine_args['pressure'] = self._pressure * PRESS_UNIT_CONVERSION
        else:
            raise NotImplementedError("routine is currently limited to 'NVE', 'NVT_Nose_Hoover', 'NPT_Nose_Hoover'")
        self._gen_fn()
        self._state = self._init_fn(jax.random.PRNGKey(0),
                                  initial_position,
                                  mass=self._mass * MASS_UNIT_CONVERSION,
                                  kT=((self._temperature if self._temperature is not None else 0)
                                      * TEMP_UNIT_CONVERSION),
                                  nbrs_nm=self._typed_nbrs.nbrs_nm if self._static_args['use_neighbor_list'] else None,
                                  box=self._initial_box,
                                )
        
        if initial_velocity is not None:
            self._state = self._state.set(velocity=initial_velocity)
        self._inner_step_fn = self._get_inner_step()
        self._report_fn = self._generate_report_fn()
        self.log = []

    def _gen_fn(self):
        ''' 
            Generate the energy function and relevant functions.
            Need to regenerate when static_args changes.
        '''
        self._energy_fn = self._get_energy_fn()
        
        if self._use_model_deviation:
            self._deviation_energy_fns = [
                self._get_energy_fn(load_model(path)) for path in self.model_deviation_paths
            ]
        self._init_fn, self._apply_fn = self._routine_fn(self._energy_fn,
                                                     self._shift_fn,
                                                     self._dt,
                                                     **self._routine_args)
        self._inner_step_fn = self._get_inner_step()
        self._report_fn = self._generate_report_fn()
        

    def _get_energy_fn(self, model_and_variables=None):

        if model_and_variables is None:
            model_and_variables = (self._model, self._variables)

        def energy_fn(coord, box, nbrs_nm, perturbation=1.):
            '''
                Energy function that can be used in jax_md.simulate routines.
                You can customize the energy function here, i.e. if you want to add perturbations.
            '''
            # Atoms are reordered and grouped by type in neural network inputs.
            coord = coord[self._type_idx.argsort(kind='stable')] * perturbation
            box = box * perturbation
            # Ensure coord and box is replicated on all devices
            sharding = jax.sharding.PositionalSharding(jax.devices()).replicate()
            coord = jax.lax.with_sharding_constraint(coord, sharding)
            if box.size == 1:
                box = box * jnp.eye(3)
            elif box.shape == (3,):
                box = jnp.diag(box)
            box = jax.lax.with_sharding_constraint(box, sharding)

            if True: # model type is energy
                E = self._model.apply(self._variables, coord, box, self._static_args, nbrs_nm)[0]
            return E

        return jax.jit(energy_fn)

    def _check_if_use_neighbor_list(self):
        '''
            Neighbor list currently only allowed for a sufficiently big orthorhombic box.
        '''
        if self._current_box.shape == (3,3):
            return False
        else:
            return False
            return self._current_box.min() > 2 * (self._model.params['rcut'] + self._dr_buffer_neighbor)

    def _get_static_args(self, position, use_neighbor_list=True):
        '''
            Returns a FrozenDict of the complete set of static arguments for jit compilation.
        '''
        use_neighbor_list *= self._check_if_use_neighbor_list()
        if use_neighbor_list:
            static_args = nn.FrozenDict({'type_count':self._type_count, 'use_neighbor_list':True})
            return static_args
        else:
            # For npt, include extra buffer for lattice selection !! not implemented yet
            box = jnp.diag(self._current_box) if self._current_box.shape == (3,) else self._current_box
            lattice_args = compute_lattice_candidate(box[None], self._model.params['rcut'])
            static_args = nn.FrozenDict({'type_count':self._type_count, 'lattice':lattice_args, 'use_neighbor_list':False})
            return static_args

    def _generate_report_fn(self):
        '''
            Returns a set of functions that reports the current state.
            You can customize report functions with the same signature.
        '''
        self._reporters = {
            "Temperature": lambda state, _: jax_md.quantity.temperature(
                                                            velocity=state.velocity,
                                                            mass=state.mass,
                                                        ) / TEMP_UNIT_CONVERSION,
            "KE": lambda state, _: jax_md.quantity.kinetic_energy(
                                                    velocity=state.velocity,
                                                    mass=state.mass,
                                                ),
            "PE": lambda state, nbrs_nm: self._energy_fn(
                                    state.position,
                                    box=state.box if "NPT" in self._routine else self._initial_box,
                                    nbrs_nm=nbrs_nm,
                                ),                 
            }
        
        if self._routine == "NVE":
            self._reporters["Invariant"] = lambda state, nbrs_nm: \
                                            self._reporters["KE"](state, nbrs_nm) + \
                                            self._reporters["PE"](state, nbrs_nm)
        elif self._routine == "NVT_Nose_Hoover":
            self._reporters["Invariant"] = lambda state, nbrs_nm: jax_md.simulate.nvt_nose_hoover_invariant(
                                            self._energy_fn,
                                            state,
                                            self._temperature,
                                            box=self._initial_box,
                                            nbrs_nm=nbrs_nm,
                                        )
        elif self._routine == "NPT_Nose_Hoover":
            self._reporters["Invariant"] = lambda state, nbrs_nm: jax_md.simulate.npt_nose_hoover_invariant(
                                            self._energy_fn,
                                            state,
                                            self._temperature,
                                            self._pressure,
                                            box=state.box,
                                            nbrs_nm=nbrs_nm,
                                        )
        if self._use_model_deviation:
            self._reporters["Model_Devi"] = lambda state, nbrs_nm: jnp.array([
                -jax.grad(fn)(
                    state.position,
                    box=state.box if "NPT" in self._routine else self._initial_box,
                    nbrs_nm=nbrs_nm,
                )
                for fn in self._deviation_energy_fns
            ]).std(axis=0).max()

        def report_fn(state, nbrs_nm):
            return [fn(state, nbrs_nm) for fn in self._reporters.values()]
        return jax.jit(report_fn)
    
    def _print_report(self):
        '''
            Print a report of the current state.
            All reports are accumulated in self.log.
        '''
        char_space = ["8"] + ["12"] * len(self._reporters) + ["6"]
        if self._is_initial_state:
            self._tic = time()
            headers = ["Step"] + list(self._reporters.keys()) + ["Time"]
            if self._debug and self._static_args['use_neighbor_list']:
                headers += ["NbrUpdateRate"]
                char_space += ["6"]
            print(" ".join([f"{header:>{char_space[i]}}" for i, header in enumerate(headers)]))
        report = [self.step] + self._report_fn(
                                    self._state,
                                    self._typed_nbrs.nbrs_nm if self._static_args['use_neighbor_list'] else None
                                ) + [time() - self._tic]
        if self._debug and self._static_args['use_neighbor_list']:
            if self._neighbor_update_profile[0] > 0:
                report += [self._neighbor_update_profile[1] / (self._neighbor_update_profile[0])]
            else:
                report += [-1]
        self.log.append(report)
        print(f"{report[0]:8d} " + " ".join([f"{value:>{char_space[i]}.3f}" for i, value in enumerate(report[1:])]))
        self._tic = time()

    def _construct_nbr_and_nbr_fn(self, position=None):
        '''
            Initial construction, or reconstruction when dr_buffer_neighbor or neighbor_buffer_ratio changes.
        '''
        if position is None:
            position = self._state.position
        self._typed_nbr_fn = typed_neighbor_list(self._current_box,
                                                self._type_idx,
                                                self._model.params['rcut'] + self._dr_buffer_neighbor,
                                                self._neighbor_buffer_ratio)
        self._typed_nbrs = self._typed_nbr_fn.allocate(position, box=self._current_box)
    
    def _check_hard_overflow(self, state, typed_nbrs):
        '''
            Check if there is any overflow that requires recompilation. Jit-compatible.
            Error code is a binary combination:
                1: Box shrunk too much with nbrs: Increase self._dr_buffer_neighbor
                2: Neighbor list buffer overflow: Reallocate neighbor list
                4: Neighbor list buffer overflow the previous step/chunk
                8: Lattice overflow: Increase lattice candidate/neighbor count
        '''
        error_code = 0
        if self._static_args['use_neighbor_list']:
            if "NPT" in self._routine:
                error_code += (1 - state.box[0] / typed_nbrs.reference_box[0]) > \
                                0.75 * self._dr_buffer_neighbor / self._model.params['rcut']
            error_code += 2 * typed_nbrs.did_buffer_overflow
        return error_code

    def _resolve_error_code(self):
        '''
            If error code is not 0 or 4, resolve the overflow and return 0 or 4.
            In this case it is not jit compatible, and regenerates arguments/functions with static information.
            If error code is 0 or 4, return 0.
        '''
        FLAG_4 = False
        if self._error_code & 1: # Box shrunk too much
            self._dr_buffer_neighbor += 0.4
            if self._check_if_use_neighbor_list():
                self._construct_nbr_and_nbr_fn()
            else:
                self._static_args = self._get_static_args(self._state.position, use_neighbor_list=False)
        elif self._error_code & 2: # Need to reallocate neighbor list
            if self._error_code & 4: # Neighbor list buffer overflow twice
                self._neighbor_buffer_ratio += 0.05
                self._construct_nbr_and_nbr_fn()
                self._typed_nbrs = self._typed_nbr_fn.allocate(self._state.position, box=self._current_box)
            else: # Neighbor list buffer overflow once, simply reallocate but mark error_code = 4
                self._typed_nbrs = self._typed_nbr_fn.allocate(self._state.position, box=self._current_box)
                FLAG_4 = True
        elif self._error_code & 8: # Not fully implemented yet!!!
            self._static_args = self._get_static_args(self._state.position, use_neighbor_list=False)

        # After resolving the error, reset the error code and regenerate functions
        if not (self._error_code == 0 or self._error_code == 4):
            self._gen_fn()
        self._error_code = 4 if FLAG_4 else 0

    def _keep_nbr_or_lattice_up_to_date(self):
        '''
            After a run finishes, the state is still one step ahead of the neighbor list or lattice.
            Call this function to ensure that the neighbor list or lattice is up-to-date.
        '''
        self._typed_nbrs = None
        if self._static_args['use_neighbor_list']:
            self._typed_nbrs, _ = self._soft_update_nbrs(self._typed_nbrs,
                                                      self._state.position,
                                                      self._current_box,
                                                      self._neighbor_update_profile)
            self._error_code = self._check_hard_overflow(self._state, self._typed_nbrs)
        else:
            self._error_code = self._check_hard_overflow(self._state, None)
        self._resolve_error_code()

    def _soft_update_nbrs(self, typed_nbrs, position, box, profile):
        '''
            A lazy jit-compatible update of neighbor list.
            Intuition: only update if atoms have moved more than dr_buffer_neighbor/2.
        '''
        scale = typed_nbrs.reference_box[0] / box[0]
        scaled_position = (position % box) * scale
        rcut, dr = self._model.params['rcut'], self._dr_buffer_neighbor
        safe_scale = scale * 1.02 if "NPT" in self._routine else scale
        allowed_movement = ((rcut + dr) - rcut * safe_scale) / 2
        max_movement = norm_ortho_box(
                                scaled_position - typed_nbrs.reference_position,
                                typed_nbrs.reference_box
                            ).max()
        update_required = max_movement > allowed_movement
        profile += jnp.array([1, update_required])
        typed_nbrs = jax.lax.cond(update_required,
                            jax.jit(lambda typed_nbrs: self._typed_nbr_fn.update(position, typed_nbrs, box=box)),
                            lambda typed_nbrs: typed_nbrs,
                            typed_nbrs)
        return typed_nbrs, profile
    
    def _get_inner_step(self):
        '''
            Returns a function that performs a single simulation step.
        '''
        def inner_step(states, _):
            state, typed_nbrs, error_code, profile = states
            current_box = state.box if "NPT" in self._routine else self._initial_box

            # soft update neighbor list before a step
            if self._static_args['use_neighbor_list']:
                typed_nbrs, profile = self._soft_update_nbrs(typed_nbrs, state.position, current_box, profile)

            # check if there is any hard overflow
            error_code = error_code | self._check_hard_overflow(state, typed_nbrs)

            # apply the simulation step                  
            state = self._apply_fn(
                                state,
                                box=current_box,
                                nbrs_nm=typed_nbrs.nbrs_nm if typed_nbrs else None,
                            )
            
            return ((state, typed_nbrs, error_code, profile),
                    (state.position,
                     state.velocity, 
                     state.box if "NPT" in self._routine else self._initial_box,
                    ))
        
        return inner_step

    def _initialize_run(self):
        '''
            Reset trajectory for each new run; 
            if the simulation has not been run before, include the initial state.
            Initialize run variables.
        '''
        if self._is_initial_state:
            self._position_trajectory = [self._state.position[None]]
            self._velocity_trajectory = [self._state.velocity[None]]
            self._box_trajectory = [self._current_box[None]]
        else:
            self._position_trajectory = []
            self._velocity_trajectory = []
            self._box_trajectory = []
        self._tic_of_this_run = time()
        self._tic = time()
        self._error_code = 0
        self._print_report()
        self._is_initial_state = False

    def run(self, steps):
        '''
            Run the simulation for a number of steps.
        '''

        self._initialize_run()
        remaining_steps = steps

        while remaining_steps > 0:

            # run the simulation for a jit-compiled chunk of steps 
            next_chunk = min(self.report_interval - self.step % self.report_interval,
                             self._step_chunk_size - self.step % self._step_chunk_size)
            states = (self._state,
                      self._typed_nbrs if self._static_args['use_neighbor_list'] else None,
                      self._error_code,
                      self._neighbor_update_profile)
            states_new, traj = jax.lax.scan(self._inner_step_fn,
                                            states,
                                            length=next_chunk)
            state_new, typed_nbrs_new, error_code, profile = states_new
            self._error_code |= error_code

            # If there is any hard overflow, we have to re-run the chunk
            if not (self._error_code == 0 or self._error_code == 4): 
                self._resolve_error_code()
                continue

            # If nothing overflows, update the tracked state and record the trajectory
            self._resolve_error_code()
            self._state = state_new
            self._typed_nbrs = typed_nbrs_new
            self._neighbor_update_profile = profile
            if "NPT" in self._routine:
                self._current_box = self._state.box
            self.step += next_chunk
            remaining_steps -= next_chunk
            pos_traj, vel_traj, box_traj = traj
            self._position_trajectory.append(np.array(pos_traj))
            self._velocity_trajectory.append(np.array(vel_traj))
            self._box_trajectory.append(np.array(box_traj))

            # Report at preset regular intervals
            if self.step % self.report_interval == 0:
                self._print_report()
        
        self._print_run_profile(steps, time() - self._tic_of_this_run)
        self._keep_nbr_or_lattice_up_to_date()
        trajectory = {
            'position': np.concatenate(self._position_trajectory),
            'velocity': np.concatenate(self._velocity_trajectory),
            'box': np.concatenate(self._box_trajectory)
            }
        return trajectory
    
    def _print_run_profile(self, steps, elapsed_time):
        '''
            Print the profile of the run.
        '''
        steps_per_microsecond_per_atom = 1e-6 * steps * self._state.position.shape[0]/ (elapsed_time + 1e-6)
        nanosecond_per_day = 1e-6 * steps * self._dt * (86400 / elapsed_time)
        print('# Finished %d steps in %dh %dm %ds.' %
                    (steps, elapsed_time // 3600, (elapsed_time % 3600) // 60, elapsed_time % 60))
        print('# Performance: %.3f ns/day, %.3f step/μs/atom' % (nanosecond_per_day, steps_per_microsecond_per_atom))

    def getState(self):
        '''
            Returns the current state.
        '''
        return self._state

    def getEnergy(self):
        '''
            Returns the energy (eV) of the current state.
        '''
        return self._energy_fn(
                            self._state.position,
                            box=self._current_box,
                            nbrs_nm=self._typed_nbrs.nbrs_nm if self._static_args['use_neighbor_list'] else None,
                        )
    
    def getForce(self):
        '''
            Returns the force (eV/Å) of the current state.
        '''
        return -jax.jit(jax.grad(self._energy_fn))(
                            self._state.position,
                            box=self._current_box,
                            nbrs_nm=self._typed_nbrs.nbrs_nm if self._static_args['use_neighbor_list'] else None,
                        )

    def getPressure(self):
        '''
            Returns the pressure (bar) of the current state.
        '''
        KE = jax_md.quantity.kinetic_energy(velocity=self._state.velocity,
                                            mass=self._state.mass)
        P_measured = jax_md.quantity.pressure(
                        self._energy_fn,
                        self._state.position,
                        self._current_box,
                        KE,
                        box=self._current_box,
                        nbrs_nm=self._typed_nbrs.nbrs_nm if self._static_args['use_neighbor_list'] else None,
                    )
        return P_measured / PRESS_UNIT_CONVERSION
