from typing import List, Optional

import jax_md
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import gc
from time import time
from jax.sharding import PartitionSpec as PSpec

from .data import compute_lattice_candidate
from .utils import (load_model, norm_ortho_box, get_p3mlr_fn, reorder_by_device, dplr_charges)
from .md_utils import (typed_neighbor_list, min_image_unwrap, print_run_profile,
            prepare_dumps, write_dump_frame, stack_typed_nbrs_per_bead, iso_pressure)
# Back-compat re-export; the ASE calculator now lives in ase_calc.py.
from .ase_calc import DPJaxCalculator  # noqa: F401
from .routine import *
from .routine import _parse_couple_axes
from . import pimd as _pimd
from functools import partial

MASS_UNIT_CONVERSION = 1.036427e2 # from Dalton to eV * fs^2 / Å^2
TEMP_UNIT_CONVERSION = 8.617333e-5 # from Kelvin to eV
PRESS_UNIT_CONVERSION = 6.241509e-7 # from bar to eV / Å^3


class Simulation:
    '''
        A deepmd-based simulation class that wraps jax_md.simulate routines.
        Two-step usage: instantiate and run(steps).
    '''

    step: int = 0
    _step_chunk_size: int = 200
    _dr_buffer_lattice: float = 1.
    _neighbor_update_profile: jax.Array = jnp.array([0.])
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
                 report_interval=200,
                 temperature=None,
                 pressure=None,
                 debug=False,
                 seed=0,
                 model_deviation_paths=[],
                 use_neighbor_list_when_possible=True,
                 neighbor_skin: float = None,
                 neighbor_buffer_ratio: float = 1.2,
                 remove_com_motion=False,
                 tau_t=100.,
                 tau_p=1000.,
                 chain_length_t=3,
                 chain_length_p=3,
                 chain_steps_t=1,
                 chain_steps_p=1,
                 sy_steps_t=1,
                 sy_steps_p=1,
                 fixed_indices: Optional[List[int]] = None,
                 couple_axes: tuple = ((0, 1, 2),),
                 n_bead=1,
                 type_symbols: Optional[List[str]] = None,
                 pimd_lambda=1.0,
                 pimd_print_invariant=False,
                ):
        '''
            Initialize a Simulation instance.
            model_path: path to the deepmd model
            box: box size, scalar or (3,) or (3,3)
            type_idx: list of atom types, length = n_atoms. If the loaded model stores
                model.params['chemical_types'] (i.e. was trained from extxyz or with chemical_types
                supplied), type_idx is interpreted as atomic numbers (Z); otherwise as integer
                type indices in the range [0, ntypes).
            mass: atomic mass for each type, length = number of atom types
            routine: 'NVE', 'NVT', 'NVT_langevin', or 'NPT'. 'NVT'/'NPT' use Nose-Hoover; 'NVT_langevin' wraps jax_md's BAOAB Langevin thermostat (friction = 1/tau_t). When n_bead > 1 and routine='NVT', the integrator switches to path-integral MD (ring-polymer BAOAB with PILE-L thermostat).
            dt: time step size (fs)
            initial_position: shape = (n_atoms, 3)
            initial_velocity: shape = (n_atoms, 3); if None, use temperature to generate; if temperature is also None, use 0.
            report_interval: print log every report_interval steps
            temperature: temperature in NVT/NPT simulations, or NVE initial_velocity generation if initial_velocity is None.
            pressure: pressure in NPT simulations
            debug: print additional debug information
            seed: random seed for jax random number generator, e.g. in velocity initialization
            model_deviation_paths: a list of deepmd models for deviation calculation used in active learning
            use_neighbor_list_when_possible: if False, neighbor list will be disabled (for debug use)
            neighbor_skin: buffer radius for neighbor list
            neighbor_buffer_ratio: capacity of neighbor list = neighbor_buffer_ratio * max_neighbors
            remove_com_motion: explicitly remove center-of-mass velocity at each step; usually unnecessary for DP models; may be useful for DPLR models
            tau: Nose-Hoover thermostat/barostat relaxation time (fs)
            chain_length: Nose-Hoover thermostat/barostat chain length
            chain_steps: Nose-Hoover thermostat/barostat chain steps
            sy_steps: Nose-Hoover thermostat/barostat number of Suzuki-Yoshida steps (must be 1,3,5,7)
            type_symbols: chemical symbols per atom type (e.g. ['O','H']); required only to use dump_* kwargs on run() since XYZ output needs element labels.
            fixed_indices: zero-based indices of atoms that should remain fixed (constrained position) in the simulation.
            couple_axes: NPT-only partition of a subset of {0,1,2} into coupled groups. Each inner tuple lists axes sharing one scale factor; axes not listed are held fixed. Examples: ((0,1,2),) iso (default); ((0,1),) semi-iso (xy couple, z fixed); ((0,),) uniaxial (only x moves); ((0,),(1,),(2,)) fully anisotropic ortho; ((0,1),(2,)) xy couple + z independent.
            n_bead: number of ring-polymer beads. 1 (default) = classical MD. n_bead > 1 switches routine='NVT' to path-integral MD using PILE-L thermostat (internal modes critically damped, centroid friction = 1/tau_t). initial_position may be (N,3) (replicated across beads) or (P,N,3).
            pimd_lambda: PILE internal-mode damping scale for PIMD, where gamma_k = 2 * pimd_lambda * omega_k for non-centroid modes. The centroid friction remains 1/tau_t. Default 1.0 preserves standard PILE behavior.
            pimd_print_invariant: if True, include the PIMD invariant reporter. Disabled by default because the finite-step diagnostic can drift more visibly as n_bead increases.
            ############################
            Usage:
                sim = Simulation(...)
                trajectory = sim.run(steps)
        '''
        initial_position = jnp.array(initial_position) * jnp.ones(1) # Ensure default precision
        self.report_interval = int(report_interval)
        self.log = []
        self._n_bead = int(n_bead)
        self._is_pimd = self._n_bead > 1
        self._pimd_lambda = float(pimd_lambda)
        self._pimd_print_invariant = bool(pimd_print_invariant)
        if self._pimd_lambda < 0:
            raise ValueError("pimd_lambda must be non-negative.")
        if self._is_pimd:
            if routine not in ('NVT', 'NPT'):
                raise NotImplementedError(
                    "Path-integral MD (n_bead > 1) currently supports routine='NVT' or 'NPT'.")
            if fixed_indices is not None:
                raise NotImplementedError("fixed_indices is not supported with n_bead > 1.")
            if len(model_deviation_paths) > 0:
                raise NotImplementedError("model_deviation_paths is not supported with n_bead > 1.")
            if remove_com_motion:
                raise NotImplementedError("remove_com_motion is not supported with n_bead > 1.")
            # Accept (N,3) -> replicate to (P,N,3); or (P,N,3) used verbatim.
            if initial_position.ndim == 2:
                initial_position = jnp.broadcast_to(
                    initial_position, (self._n_bead,) + initial_position.shape
                ).astype(initial_position.dtype)
            elif initial_position.ndim == 3:
                if initial_position.shape[0] != self._n_bead:
                    raise ValueError(
                        f"initial_position leading axis {initial_position.shape[0]} "
                        f"does not match n_bead={self._n_bead}.")
            else:
                raise ValueError(
                    "initial_position must have shape (N,3) or (P,N,3).")
            self._natoms = initial_position.shape[1]
            self._temperature = temperature
            self._nm_freqs, self._nm_trans = _pimd.normal_mode_transform(
                self._n_bead, temperature * TEMP_UNIT_CONVERSION)
        else:
            self._natoms = initial_position.shape[0]
            self._temperature = temperature
        self._dt = dt
        self._routine = routine
        self._model, self._variables = load_model(model_path)
        type_idx_np = np.array(type_idx).astype(int)
        ct = self._model.params.get('chemical_types')
        if ct is not None:
            unknown = set(type_idx_np.tolist()) - set(ct)
            if unknown:
                raise ValueError('Atomic numbers %s in type_idx are not in model.params["chemical_types"]=%s'
                                 % (sorted(unknown), ct))
            z_to_idx = {z: i for i, z in enumerate(ct)}
            type_idx_np = np.array([z_to_idx[z] for z in type_idx_np], dtype=int)
        self._type_idx = type_idx_np
        self._mass = jnp.array(np.array(mass)[np.array(self._type_idx)]) # AMU
        if type_symbols is not None:
            n_types_needed = int(self._type_idx.max()) + 1
            if len(type_symbols) < n_types_needed:
                raise ValueError(
                    f"type_symbols has length {len(type_symbols)} but type_idx "
                    f"references types up to {n_types_needed - 1}.")
        self._type_symbols = list(type_symbols) if type_symbols is not None else None
        type_count = np.bincount(self._type_idx)
        self._type_count = np.pad(type_count, (0, self._model.params['ntypes'] - len(type_count)))
        self._debug = debug
        # Chunk memory budget ~ chunk_size * n_bead * natoms.
        self._step_chunk_size = max(10, min(100, 100000 // (self._natoms * self._n_bead)))
        self._model_deviation_paths = model_deviation_paths
        self._use_model_deviation = len(model_deviation_paths) > 0
        self._remove_com_motion = remove_com_motion
        if neighbor_skin is None:
            self._neighbor_skin = 0.5 if "NPT" in self._routine else 0.3
        else:
            self._neighbor_skin = neighbor_skin
        self._neighbor_buffer_ratio = neighbor_buffer_ratio
        # If orthorhombic, keep box as (3,); else keep (3,3)
        box = jnp.array(box)
        if box.size == 1:
            self._initial_box = box.item() * jnp.ones(3)
        elif box.shape == (3,3) and (box == jnp.diag(jnp.diag(box))).all():
            self._initial_box = jnp.diag(box)
        else:
            self._initial_box = box
        self._current_box = self._initial_box
        self._static_args = self._get_static_args(initial_position, use_neighbor_list_when_possible)
        self._displacement_fn, shift = jax_md.space.periodic_general(self._initial_box, fractional_coordinates=False)

        # Create mask for constraining atoms
        mobile = np.ones(self._natoms)
        if fixed_indices is not None:
            mobile[fixed_indices] = 0
            self._mobile = jnp.array(mobile, dtype=bool)
        else: 
            self._mobile = None

        if self._mobile is not None and self._routine not in ('NVT', 'NVE', 'NVT_langevin'):
            raise NotImplementedError("Fixing atoms currently limited to NVE, NVT, and NVT_langevin routines.")

        def _shift_fn_wrapper(x, dx, **kwargs):
            if self._mobile is not None:
                return jnp.where(mobile[:, None], shift(x, dx, **kwargs), x)
            return shift(x, dx, **kwargs)
        self._shift_fn = _shift_fn_wrapper

        # Defaults; overridden in the NPT branch below.
        self._couple_axes = ((0, 1, 2),)
        self._moving_mask_np = np.array([1, 1, 1], dtype=bool)
        self._group_ids_np = np.array([0, 0, 0], dtype=np.int32)
        _is_default_couple = (isinstance(couple_axes, tuple)
                              and couple_axes == ((0, 1, 2),))
        if not _is_default_couple and "NPT" not in self._routine:
            raise ValueError("couple_axes is only meaningful for routine='NPT'")

        self._routine_fn, self._routine_args = self._build_routine(
            pressure=pressure, couple_axes=couple_axes,
            tau_t=tau_t, tau_p=tau_p,
            chain_length_t=chain_length_t, chain_length_p=chain_length_p,
            chain_steps_t=chain_steps_t, chain_steps_p=chain_steps_p,
            sy_steps_t=sy_steps_t, sy_steps_p=sy_steps_p,
        )
        init_info = []
        if self._temperature is not None and self._routine != "NVE":
            init_info.append(f"temperature={self._temperature:g} K")
        if "NPT" in self._routine:
            init_info.append(f"pressure={pressure:g} bar")
        if self._is_pimd:
            init_info.append(f"n_beads={self._n_bead}")
            init_info.append(f"pimd_lambda={self._pimd_lambda:g}")
        suffix = f" ({', '.join(init_info)})" if init_info else ""
        print(f"# Initialized {self._routine} simulation with {self._natoms} atoms{suffix}")
        if self._is_pimd and self._pimd_print_invariant:
            print("# Warning: PIMD invariant is a finite-step diagnostic and may drift more visibly as n_beads increases.")

        # Initialize neighbor list if needed
        if self._static_args['use_neighbor_list']:
            self._construct_nbr_and_nbr_fn(initial_position)
        self._gen_fn()
        self._state = self._init_fn(
                                jax.random.PRNGKey(seed),
                                initial_position,
                                mass=self._mass * MASS_UNIT_CONVERSION,
                                kT=((self._temperature if self._temperature is not None else 0)
                                      * TEMP_UNIT_CONVERSION),
                                nbrs_nm=self._nbrs_nm,
                                box=self._initial_box,
                            )

        if initial_velocity is not None:
            self.setVelocity(initial_velocity)

    def _gen_fn(self):
        '''
            Generate the energy function and relevant functions.
            Need to regenerate when static_args changes.
            Beware of the order of functions here, as some functions depend on previous generated functions.
            Neighbor list functions are generated in _construct_nbr_and_nbr_fn separately.
        '''
        self._energy_fn = self._get_energy_fn()
        self._force_fn = lambda coord, **kwargs: -jax.grad(self._energy_fn)(coord, **kwargs)

        def pressure_fn(state, box, nbrs_nm):
            kT = self._temperature * TEMP_UNIT_CONVERSION
            return iso_pressure(self._energy_fn, state.position, box, self._natoms, kT,
                                nbrs_nm=nbrs_nm) / PRESS_UNIT_CONVERSION
        self._pressure_fn = pressure_fn
        if self._use_model_deviation:
            self._deviation_energy_fns = [
                self._get_energy_fn(load_model(path)) for path in self._model_deviation_paths
            ]
        self._init_fn, self._apply_fn = self._routine_fn(self._energy_fn,
                                                         self._shift_fn,
                                                         self._dt,
                                                         **self._routine_args)

        self._report_fn = self._generate_report_fn()
        self._soft_update_nbrs = self._get_soft_update_nbrs_fn()
        self._check_hard_overflow = self._get_check_hard_overflow_fn()
        self._multiple_inner_step_fn = self._get_inner_step()

    def _build_routine(self, *, pressure, couple_axes, tau_t, tau_p,
                       chain_length_t, chain_length_p,
                       chain_steps_t, chain_steps_p,
                       sy_steps_t, sy_steps_p):
        '''Pick integrator function and its kwargs for self._routine. NPT
        branches additionally set _couple_axes / _moving_mask_np /
        _group_ids_np / _pressure on self.'''
        kT = (self._temperature * TEMP_UNIT_CONVERSION
              if self._temperature is not None else None)
        if "NPT" in self._routine:
            if pressure is None:
                raise ValueError("Missing argument 'pressure' (in bar) for routine 'NPT'")
            groups, n_per_group_np, self._group_ids_np, membership_np = _parse_couple_axes(couple_axes)
            self._couple_axes = groups
            self._moving_mask_np = (self._group_ids_np >= 0)
            self._pressure = pressure

        key = (self._routine, self._is_pimd)
        if key == ("NVE", False):
            return nve, {'mobile_mask': self._mobile}
        if key == ("NVT", False):
            return nvt_nose_hoover, {
                'kT': kT, 'tau': tau_t,
                'chain_length': chain_length_t,
                'chain_steps': chain_steps_t,
                'sy_steps': sy_steps_t,
                'mobile_mask': self._mobile,
            }
        if key == ("NVT", True):
            # PILE-L: internal modes gamma_k = 2 * pimd_lambda * omega_k; centroid friction = 1/tau_t.
            gamma = 2.0 * self._pimd_lambda * self._nm_freqs
            gamma[0] = 1.0 / tau_t
            return _pimd.nvt_langevin_pimd, {
                'kT': kT,
                'box': self._current_box,
                'gamma_P': jnp.asarray(gamma),
                'nm_trans': jnp.asarray(self._nm_trans),
                'nm_freqs': jnp.asarray(self._nm_freqs),
            }
        if key == ("NVT_langevin", False):
            return nvt_langevin, {
                'kT': kT, 'gamma': 1.0 / tau_t,
                'mobile_mask': self._mobile,
            }
        if key == ("NPT", False):
            return npt_nose_hoover, {
                'pressure': pressure * PRESS_UNIT_CONVERSION,
                'kT': kT,
                'barostat_kwargs': {'tau': tau_p, 'chain_length': chain_length_p,
                                    'chain_steps': chain_steps_p, 'sy_steps': sy_steps_p},
                'thermostat_kwargs': {'tau': tau_t, 'chain_length': chain_length_t,
                                      'chain_steps': chain_steps_t, 'sy_steps': sy_steps_t},
                'couple_axes': self._couple_axes,
            }
        if key == ("NPT", True):
            gamma_P = 2.0 * self._pimd_lambda * self._nm_freqs
            gamma_P[0] = 1.0 / tau_t
            return _pimd.npt_langevin_pimd, {
                'pressure': pressure * PRESS_UNIT_CONVERSION,
                'kT': kT,
                'gamma_P': jnp.asarray(gamma_P),
                'gamma_box': 1.0 / tau_p,
                'nm_trans': jnp.asarray(self._nm_trans),
                'nm_freqs': jnp.asarray(self._nm_freqs),
                'membership': membership_np,
                'n_per_group': n_per_group_np,
                'group_ids': self._group_ids_np,
                'tau_p': tau_p,
            }
        raise NotImplementedError(
            "routine currently limited to 'NVE', 'NVT', 'NVT_langevin', 'NPT'")

    def _get_energy_fn(self, model_and_variables=None):

        if model_and_variables is None:
            model_and_variables = (self._model, self._variables)
        model, variables = model_and_variables
        if model.params['type'] == 'dplr':
            if self._current_box.size > 3:
                raise NotImplementedError("dplr model currently only supports orthorhombic box")
            wc_model, wc_variables = model.params['dplr_wannier_model_and_variables']
            p3mlr_fn = get_p3mlr_fn(
                            self._current_box,
                            model.params['dplr_beta'],
                            resolution=model.params['dplr_resolution'],
                        )
            qatoms_np, qwc_np = dplr_charges(
                self._type_idx, model.params['dplr_q_atoms'], model.params['dplr_q_wc'],
                wc_model.params['nsel'], self._model.params['ntypes'])
            qatoms, qwc = jnp.array(qatoms_np), jnp.array(qwc_np)

        def _single_conf_energy(coord, box, nbrs_nm):
            '''Evaluate the ML (and optional DPLR) potential on one configuration.'''
            E = model.apply(variables, coord, box, self._static_args, nbrs_nm)[0]
            if model.params['type'] == 'dplr':
                wc = wc_model.wc_predict(wc_variables, coord, box, self._static_args, nbrs_nm)
                E = E + p3mlr_fn(
                    jnp.concatenate([coord, wc]),
                    jnp.concatenate([qatoms, qwc]),
                    jnp.diag(box),
                )
            return E

        def energy_fn(coord, nbrs_nm, perturbation=1., **kwargs):
            '''
                ML (and optional DPLR) potential energy.
                coord is (N, 3) normally, or (P, N, 3) under PIMD in which case
                the result is the bead-averaged potential <V>_beads.
            '''
            # if box in kwargs, use it else self._current_box, and convert to (3,3)
            box = kwargs['box'] if 'box' in kwargs else self._current_box
            if box.size == 1:
                box = box * jnp.eye(3)
            elif box.shape == (3,):
                box = jnp.diag(box)
            if self._mobile is not None:
                coord = jnp.where(self._mobile[:, None], coord,
                                  jax.lax.stop_gradient(coord))
            # perturbation = 1, required by jax-md pressure calculation
            coord = coord * perturbation
            box = box * perturbation
            coord = jax.lax.with_sharding_constraint(coord, PSpec())
            box = jax.lax.with_sharding_constraint(box, PSpec())

            if not self._is_pimd:
                return _single_conf_energy(coord, box, nbrs_nm)
            else:
                nbrs_axes = None if nbrs_nm is None else 0
                E_per_bead = jax.vmap(_single_conf_energy, in_axes=(0, None, nbrs_axes))(coord, box, nbrs_nm) # (P,)
                return jnp.sum(E_per_bead) / self._n_bead # average over beads

        return energy_fn

    @property
    def _nbrs_nm(self):
        return self._typed_nbrs.nbrs_nm if self._static_args['use_neighbor_list'] else None

    def _check_if_use_neighbor_list(self):
        '''
            Neighbor list currently only allowed for a sufficiently big orthorhombic box.
        '''
        if self._current_box.shape == (3,3):
            return False
        return self._current_box.min() > 2 * (self._model.params['rcut'] + self._neighbor_skin) * (1.02 if "NPT" in self._routine else 1.)

    def _get_static_args(self, position, use_neighbor_list=True):
        '''
            Returns a FrozenDict of the complete set of static arguments for jit compilation.
        '''
        use_neighbor_list *= self._check_if_use_neighbor_list()
        if use_neighbor_list:
            static_args = nn.FrozenDict({'type_idx':tuple(self._type_idx), 'use_neighbor_list':True})
            return static_args
        else:
            # For npt, include extra buffer for lattice selection !! not implemented yet
            box = jnp.diag(self._current_box) if self._current_box.shape == (3,) else self._current_box
            lattice_args = compute_lattice_candidate(box[None], self._model.params['rcut'])
            static_args = nn.FrozenDict({'type_idx':tuple(self._type_idx), 'lattice':lattice_args, 'use_neighbor_list':False})
            return static_args

    def _generate_report_fn(self):
        '''
            Returns a set of functions that reports the current state.
            You can customize report functions with the same signature.
        '''
        _temp_dof_scale = self._natoms / int(np.sum(self._mobile)) if self._mobile is not None else 1
        self._reporters = {
            "Temperature": lambda state, _: jax_md.quantity.temperature(
                                                            velocity=state.velocity,
                                                            mass=state.mass,
                                                        ) / TEMP_UNIT_CONVERSION * _temp_dof_scale / self._n_bead,
            "KE": lambda state, _: jax_md.quantity.kinetic_energy(
                                                    velocity=state.velocity,
                                                    mass=state.mass,
                                                ) / self._n_bead**2,
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
        elif self._routine == "NVT" and not self._is_pimd:
            self._reporters["Invariant"] = lambda state, nbrs_nm: nvt_nose_hoover_invariant(
                                            self._energy_fn,
                                            state,
                                            self._temperature * TEMP_UNIT_CONVERSION,
                                            nbrs_nm=nbrs_nm,
                                        )
        elif self._routine == "NVT_langevin":
            self._reporters["Invariant"] = lambda state, nbrs_nm: (
                jax_md.simulate.kinetic_energy(state)
                + self._energy_fn(state.position, box=self._initial_box, nbrs_nm=nbrs_nm)
                - state.thermostat_work
            )
        elif self._routine == "NVT" and self._is_pimd:
            if self._pimd_print_invariant:
                omega_P = self._n_bead * self._temperature * TEMP_UNIT_CONVERSION / _pimd.HBAR
                def _pimd_nvt_invariant(state, nbrs_nm):
                    H_rp = (jax_md.simulate.kinetic_energy(state)
                            + self._n_bead * self._energy_fn(
                                state.position, box=self._initial_box, nbrs_nm=nbrs_nm)
                            + _pimd.spring_energy(
                                state.position, state.mass, self._initial_box, omega_P))
                    return (H_rp - state.thermostat_work) / self._n_bead
                self._reporters["Invariant"] = _pimd_nvt_invariant
        elif self._routine == "NPT":
            self._reporters["Pressure"] = lambda state, nbrs_nm: self._pressure_fn(state, state.box, nbrs_nm)
            # One column per moving axis. Fixed axes are omitted; axes in the
            # same coupled group track exactly but their absolute lengths can
            # differ when the reference box is non-cubic.
            axis_letters = ['x', 'y', 'z']
            for a in sorted(a for g in self._couple_axes for a in g):
                self._reporters[f'box_{axis_letters[a]}'] = (lambda state, _, a=a: state.box[a])
            if not self._is_pimd:
                self._reporters["Invariant"] = lambda state, nbrs_nm: npt_nose_hoover_invariant(
                                                self._energy_fn,
                                                state,
                                                self._pressure * PRESS_UNIT_CONVERSION,
                                                self._temperature * TEMP_UNIT_CONVERSION,
                                                nbrs_nm=nbrs_nm,
                                            )
            else:
                if self._pimd_print_invariant:
                    omega_P = self._n_bead * self._temperature * TEMP_UNIT_CONVERSION / _pimd.HBAR
                    P_target = self._pressure * PRESS_UNIT_CONVERSION
                    def _pimd_npt_invariant(state, nbrs_nm):
                        box = state.box
                        V = jnp.prod(box) if box.ndim == 1 else jnp.linalg.det(box)
                        KE_box = 0.5 * jnp.sum(state.box_momentum ** 2 / state.box_mass)
                        # The implemented PIMD piston acts on the physical centroid
                        # variables, so the conserved system enthalpy is the
                        # bead-averaged ring-polymer Hamiltonian.
                        H_rp = (jax_md.simulate.kinetic_energy(state)
                                + self._n_bead * self._energy_fn(
                                    state.position, box=box, nbrs_nm=nbrs_nm)
                                + _pimd.spring_energy(
                                    state.position, state.mass, box, omega_P))
                        return ((H_rp - state.thermostat_work) / self._n_bead
                                + KE_box + P_target * V
                                - state.box_thermostat_work)
                    self._reporters["Invariant"] = _pimd_npt_invariant
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
        # Box-length columns only need ~6 chars ("12.487"); give them less width.
        char_space = ["8"] + ["7" if k.startswith('box_') else "12" for k in self._reporters] + ["6"]
        if self._is_initial_state:
            headers = ["Step"] + list(self._reporters.keys()) + ["Time"]
            if self._debug and self._static_args['use_neighbor_list']:
                headers += ["NbrUpdateRate"]
                char_space += ["6"]
            print(" ".join([f"{header:{char_space[i]}}" for i, header in enumerate(headers)]))
        report = [self.step] + self._report_fn(
                                    self._state,
                                    self._nbrs_nm
                                ) + [time() - self._tic_between_report]
        if self._debug and self._static_args['use_neighbor_list']:
            char_space += ["6"]
            report += [self._neighbor_update_profile.item()]
        self.log.append(report)
        print(f"{report[0]:<8d} " + " ".join([f"{value:<{char_space[i+1]}.3f}" for i, value in enumerate(report[1:])]))
        self._tic_between_report = time()

    def _construct_nbr_and_nbr_fn(self, position=None):
        '''
            Initial construction, or reconstruction when dr_buffer_neighbor or
            neighbor_buffer_ratio changes. Under PIMD, replicas are stacked
            along a leading P axis (see stack_typed_nbrs_per_bead).
        '''
        self._typed_nbr_fn = typed_neighbor_list(self._current_box,
                                                 self._type_idx,
                                                 self._type_count,
                                                 self._model.params['rcut'] + self._neighbor_skin,
                                                 self._neighbor_buffer_ratio)
        pos = self._state.position if position is None else position
        if self._is_pimd:
            self._typed_nbrs = stack_typed_nbrs_per_bead(
                self._typed_nbr_fn, pos, self._current_box, self._n_bead)
        else:
            self._typed_nbrs = self._typed_nbr_fn.allocate(
                pos, box=self._current_box)

    def _get_check_hard_overflow_fn(self):

        def check_hard_overflow(state, typed_nbrs):
            '''
                Check if there is any overflow that requires recompilation. Jit-compatible.
                Error code is a binary combination:
                    1: Box shrunk too much with nbrs: Increase self._neighbor_skin or disable neighbor list
                    2: Neighbor list buffer overflow: Reallocate neighbor list
                    4: Neighbor list buffer overflow the previous step/chunk
                    8: Lattice overflow: Increase lattice candidate/neighbor count
                    16: Nan or Inf encountered in coordinates or velocities
                Under PIMD the overflow flag reduces across beads via .any().
            '''
            error_code = 0
            if self._static_args['use_neighbor_list']:
                if "NPT" in self._routine:
                    # Per-axis max shrinkage (fixed axes trivially contribute 0); per-axis min size.
                    shrink = (1 - state.box / typed_nbrs.reference_box).max()
                    error_code += ((shrink >
                                    0.8 * self._neighbor_skin / (self._model.params['rcut'] + self._neighbor_skin))
                                    | (state.box.min() < 2 * (self._model.params['rcut'] + self._neighbor_skin) * 1.02))
                error_code += 2 * (jnp.asarray(typed_nbrs.did_buffer_overflow).any() > 0)
            is_nan = jnp.isnan(state.position).any() | jnp.isnan(state.velocity).any()
            is_inf = jnp.isinf(state.position).any() | jnp.isinf(state.velocity).any()
            error_code += 16 * (is_nan | is_inf)
            return error_code
        return check_hard_overflow

    def _resolve_error_code(self):
        '''
            If error code is not 0 or 4, resolve the overflow and return 0 or 4.
            In this case it is not jit compatible, and regenerates arguments/functions with static information.
            If error code is 0 or 4, return 0.
        '''
        FLAG_4 = False
        if self._error_code & 1: # Box shrunk too much
            self._neighbor_skin += 0.4
            if self._check_if_use_neighbor_list():
                self._construct_nbr_and_nbr_fn()
            else:
                self._static_args = self._get_static_args(self._state.position, use_neighbor_list=False)
        elif self._error_code & 2: # Need to reallocate neighbor list
            if self._error_code & 4: # Neighbor list buffer overflow twice
                self._neighbor_buffer_ratio += 0.05
                self._construct_nbr_and_nbr_fn()
            else: # Neighbor list buffer overflow once, simply reallocate
                if self._is_pimd:
                    # Keep the stacked layout consistent; this reallocates every bead.
                    self._construct_nbr_and_nbr_fn()
                else:
                    self._typed_nbrs = self._typed_nbr_fn.allocate(
                        self._state.position, box=self._current_box)
            FLAG_4 = True
        elif self._error_code & 8: # Not fully implemented yet!!!
            self._static_args = self._get_static_args(self._state.position, use_neighbor_list=False)

        # After resolving the error, reset the error code and regenerate functions
        if not (self._error_code == 0 or self._error_code == 4):
            print("# Successfully resolved overflow code", self._error_code)
            self._gen_fn()
        self._error_code = 4 if FLAG_4 else 0

    def _keep_nbr_or_lattice_up_to_date(self):
        '''
            After a run finishes, the state is still one step ahead of the neighbor list or lattice.
            Call this function to ensure that the neighbor list or lattice is up-to-date.
        '''
        if self._static_args['use_neighbor_list']:
            self._typed_nbrs, _ = jax.jit(self._soft_update_nbrs)(
                self._typed_nbrs, self._state.position, self._current_box,
                self._neighbor_update_profile,
            )
            self._error_code = jax.jit(self._check_hard_overflow)(self._state, self._typed_nbrs)
        else:
            self._error_code = jax.jit(self._check_hard_overflow)(self._state, None)
        self._resolve_error_code()

    def _get_soft_update_nbrs_fn(self):
        '''
            Lazy jit-compatible neighbor list update: refresh only when atoms
            have moved more than dr_buffer_neighbor/2. Under PIMD the drift
            test reduces across beads and the stacked pytree is updated jointly.
        '''
        rcut_static = self._model.params['rcut']

        def _drift_per_bead(typed_nbrs_single, position_N3, box):
            scale = typed_nbrs_single.reference_box / box
            scaled_position = (position_N3 % box) * scale
            scaled_position = reorder_by_device(scaled_position, self._type_idx)
            return norm_ortho_box(
                scaled_position - typed_nbrs_single.reference_position,
                typed_nbrs_single.reference_box,
            ).max()

        # Vmap the per-bead drift / update only when in PIMD; otherwise these
        # are the no-op identities.
        if self._is_pimd:
            drift_fn = lambda nbrs, pos, box: jax.vmap(
                _drift_per_bead, in_axes=(0, 0, None))(nbrs, pos, box).max()
            ref_box_of = lambda nbrs: nbrs.reference_box[0]
            update_fn = lambda nbrs, pos, box: jax.vmap(
                lambda n, p: self._typed_nbr_fn.update(p, n, box=box),
                in_axes=(0, 0))(nbrs, pos)
        else:
            drift_fn = _drift_per_bead
            ref_box_of = lambda nbrs: nbrs.reference_box
            update_fn = lambda nbrs, pos, box: self._typed_nbr_fn.update(
                pos, nbrs, box=box)

        def soft_update_nbrs(typed_nbrs, position, box, profile):
            ref_box = ref_box_of(typed_nbrs)
            scale_max = (ref_box / box).max()
            safe_scale = scale_max * 1.02 if "NPT" in self._routine else scale_max
            allowed_movement = ((rcut_static + self._neighbor_skin)
                                - rcut_static * safe_scale) / 2
            max_movement = drift_fn(typed_nbrs, position, box)
            update_required = max_movement > allowed_movement - ref_box.min() * 1e-6
            profile = profile * (1 - 1/100) + 1/100 * update_required
            typed_nbrs = jax.lax.cond(
                update_required,
                lambda: update_fn(typed_nbrs, position, box),
                lambda: typed_nbrs,
            )
            return typed_nbrs, profile

        return soft_update_nbrs


    def remove_com_motion(self):
        '''
            Remove the center-of-mass velocity from the current state.
        '''
        if self._is_pimd:
            # mass is (N,1); velocity is (P,N,3). Sum over (beads, atoms) -> (3,).
            total_mom = (self._state.velocity * self._state.mass).sum(axis=(0, 1))
            total_mass = self._state.mass.sum() * self._n_bead
            velocity = self._state.velocity - total_mom / total_mass
            self.setVelocity(velocity)
            return
        mobile_mass = self._state.mass * (self._mobile[:, None] if self._mobile is not None else jnp.ones_like(self._state.mass, dtype=bool))
        velocity = self._state.velocity - (self._state.velocity * mobile_mass).sum(0) / mobile_mass.sum()
        self.setVelocity(velocity)
        
    def _get_inner_step(self):
        '''
            Returns a jitted function that performs multiple simulation steps.
        '''
        def inner_step(states, _):
            '''
                Performs a single simulation step.
            '''
            state, typed_nbrs, error_code, profile = states
            previous_state_position = state.position
            current_box = state.box if "NPT" in self._routine else self._initial_box

            # soft update neighbor list before a step
            if self._static_args['use_neighbor_list']:
                typed_nbrs, profile = self._soft_update_nbrs(
                    typed_nbrs, state.position, current_box, profile)

            # check if there is any hard overflow
            error_code = error_code | self._check_hard_overflow(state, typed_nbrs)

            # apply the simulation step
            state = self._apply_fn(
                                state,
                                nbrs_nm=typed_nbrs.nbrs_nm if typed_nbrs else None,
                            )
            box_now = state.box if "NPT" in self._routine else self._current_box
            state = state.set(position=min_image_unwrap(
                previous_state_position, state.position, box_now))
            if self._remove_com_motion:
                mobile_mass = state.mass * (self._mobile[:, None] if self._mobile is not None else jnp.ones_like(state.mass, dtype=bool))
                velocity = state.velocity - (state.velocity * mobile_mass).sum(0) / mobile_mass.sum()
                state = state.set(momentum=state.mass * velocity)
            box_now = state.box if "NPT" in self._routine else self._initial_box
            # Under PIMD, emit the centroid with a minimum-image reduction.
            centroid_frame = _pimd.centroid(state.position, box_now) if self._is_pimd else state.position
            return ((state, typed_nbrs, error_code, profile),
                    (state.position, state.velocity, box_now, centroid_frame))

        @partial(jax.jit, static_argnums=(1,))
        def multiple_inner_step(states, length):
            '''
                states = (state, typed_nbrs, error_code, profile)
            '''
            return jax.lax.scan(inner_step, states, length=length)

        return multiple_inner_step

    def _initialize_run(self, steps,
                        dump_position=None, dump_velocity=None,
                        dump_centroid=None, dump_interval=None):
        '''
            Reset trajectory for each new run; if the simulation has not been
            run before, include the initial state. When any ``dump_*`` path is
            given, set up streaming to disk instead of preallocating in-memory
            trajectory buffers, and write the initial state to file.
        '''
        self._dumps, self._write_interval = prepare_dumps(
            {'position': dump_position, 'velocity': dump_velocity,
             'centroid': dump_centroid},
            self._type_idx, self._type_symbols, self._n_bead,
            default_interval=self.report_interval, dump_interval=dump_interval)
        self._has_dumps = len(self._dumps) > 0

        print(f'# Running {steps} steps...')
        traj_length = steps + int(self._is_initial_state)
        self._offset = self.step - int(self._is_initial_state)
        traj_dtype = np.float64 if jax.config.read('jax_enable_x64') else np.float32
        if not self._has_dumps:
            # preallocate space for trajectory
            frame_shape = ((self._n_bead, self._natoms, 3) if self._is_pimd
                           else (self._natoms, 3))
            centroid_shape = (self._natoms, 3)
            natoms_eff = self._n_bead * self._natoms
            try:
                safe_buffer = np.zeros((traj_length, 5*natoms_eff, 3), dtype=traj_dtype)
                del safe_buffer
                gc.collect()
                self._position_trajectory = np.zeros((traj_length,) + frame_shape, dtype=traj_dtype)
                self._velocity_trajectory = np.zeros((traj_length,) + frame_shape, dtype=traj_dtype)
                self._box_trajectory = np.zeros((traj_length,) + self._current_box.shape, dtype=traj_dtype)
                if self._is_pimd:
                    self._centroid_trajectory = np.zeros((traj_length,) + centroid_shape, dtype=traj_dtype)
            except MemoryError:
                raise MemoryError("Trajectory too large to fit in CPU RAM. Please pass dump_position=... (and friends) to stream to disk, or split into multiple shorter runs.")
            try:
                safe_buffer = np.zeros((traj_length, 10*natoms_eff, 3), dtype=traj_dtype)
            except MemoryError:
                print("# Warning: A long trajectory may exhaust CPU RAM. Consider dump_position=... to stream to disk.")
            if self._is_initial_state:
                self._position_trajectory[0] = self.getPosition()
                self._velocity_trajectory[0] = self.getVelocity()
                self._box_trajectory[0] = self.getBox()
                if self._is_pimd:
                    self._centroid_trajectory[0] = self.getCentroidPosition()
        else:
            if self._is_initial_state and (self.step % self._write_interval == 0):
                write_dump_frame(self._dumps,
                                 self.getPosition(), self.getVelocity(),
                                 self.getCentroidPosition() if self._is_pimd else None,
                                 self._current_box)
        self._tic_of_this_run = time()
        self._tic_between_report = time()
        self._error_code = 0
        self._print_report()
        self._is_initial_state = False

    def run(self, steps,
            dump_position: Optional[str] = None,
            dump_velocity: Optional[str] = None,
            dump_centroid: Optional[str] = None,
            dump_interval: Optional[int] = None):
        '''
            Run for ``steps`` steps. Returns a trajectory dict, or ``None`` if
            any ``dump_*`` kwarg is given (frames stream to XYZ instead).
        '''
        self._initialize_run(steps,
                             dump_position=dump_position,
                             dump_velocity=dump_velocity,
                             dump_centroid=dump_centroid,
                             dump_interval=dump_interval)

        remaining_steps = steps
        while remaining_steps > 0:

            # run the simulation for a jit-compiled chunk of steps
            next_chunk = min(self.report_interval - self.step % self.report_interval,
                             self._step_chunk_size,
                             remaining_steps)
            states = (self._state,
                      self._typed_nbrs if self._static_args['use_neighbor_list'] else None,
                      self._error_code,
                      self._neighbor_update_profile)
            states_new, traj = self._multiple_inner_step_fn(states, next_chunk)
            state_new, typed_nbrs_new, error_code, profile = states_new
            self._error_code |= error_code

            if self._error_code & 16:
                print("# Warning: Nan or Inf encountered in simulation. Terminating.")
                remaining_steps = 0

            # If there is any hard overflow, we have to re-run the chunk
            if not (self._error_code == 0 or self._error_code == 4 or self._error_code == 16):
                self._resolve_error_code()
                continue

            # If nothing overflows, update the tracked state and record the trajectory
            self._keep_nbr_or_lattice_up_to_date()
            self._state = state_new
            self._typed_nbrs = typed_nbrs_new
            self._neighbor_update_profile = profile
            if "NPT" in self._routine:
                self._current_box = self._state.box
            pos_traj, vel_traj, box_traj, centroid_traj = traj
            if self._has_dumps:
                base = int(self.step)
                for i in range(next_chunk):
                    if (base + i + 1) % self._write_interval == 0:
                        write_dump_frame(
                            self._dumps,
                            np.asarray(pos_traj[i]),
                            np.asarray(vel_traj[i]),
                            np.asarray(centroid_traj[i]) if self._is_pimd else None,
                            np.asarray(box_traj[i]),
                        )
            else:
                idx_l, idx_r = self.step - self._offset, self.step - self._offset + next_chunk
                self._position_trajectory[idx_l:idx_r] = pos_traj
                self._velocity_trajectory[idx_l:idx_r] = vel_traj
                self._box_trajectory[idx_l:idx_r] = box_traj
                if self._is_pimd:
                    self._centroid_trajectory[idx_l:idx_r] = centroid_traj
            self.step += next_chunk
            remaining_steps -= next_chunk

            # Report at preset regular intervals
            if self.step % self.report_interval == 0 or remaining_steps == 0:
                self._print_report()

        self._print_run_profile(steps, time() - self._tic_of_this_run)
        self._keep_nbr_or_lattice_up_to_date()
        if self._has_dumps:
            return None
        trajectory = {
            'position': self._position_trajectory,
            'velocity': self._velocity_trajectory,
            'box': self._box_trajectory,
        }
        if self._is_pimd:
            trajectory['centroid'] = self._centroid_trajectory
        return trajectory

    def _print_run_profile(self, steps, elapsed_time):
        print_run_profile(steps, elapsed_time, self._state.position.shape[0], self._dt)

    def getPosition(self):
        '''
            Returns the current position (Å). Shape (N, 3); (P, N, 3) under PIMD.
        '''
        return np.array(self._state.position)

    def getCentroidPosition(self):
        '''
            Ring-polymer centroid position (Å), shape (N, 3); identical to
            getPosition() in classical MD. Beads are minimum-image-shifted
            onto a common periodic image before averaging.
        '''
        if self._is_pimd:
            return np.array(_pimd.centroid(self._state.position, self._current_box))
        return np.array(self._state.position)

    def setPosition(self, position, box=None):
        '''
            Set the position (Å). Shape (N, 3); optionally (P, N, 3) for PIMD.
            Box change is only allowed under NPT, via the 'box' argument.
        '''
        position = jnp.asarray(position)
        if self._is_pimd and position.ndim == 2:
            position = jnp.broadcast_to(position, self._state.position.shape)
        if position.shape != self._state.position.shape:
            raise ValueError("Position must have the same shape as the initial position, or you have to create a new Simulation instance.")
        if box is not None:
            if not "NPT" in self._routine:
                raise ValueError("Box can only be changed in NPT simulations.")
            box = jnp.array(box)
            if box.size == 1:
                box = box.item() * jnp.ones(3)
            if box.shape != self._current_box.shape:
                raise ValueError("Box must have the same shape as the initial box.")
            # assert isotropic fluctuations
            scale = box[0] / self._current_box[0]
            if not jnp.allclose(box, self._current_box * scale, rtol=1e-4, atol=1e-6):
                raise ValueError("Only isotropic box fluctuations are allowed in the current implementation.")
            self._current_box = jnp.array(box)
        self._state = self._state.set(position=position)
        self._keep_nbr_or_lattice_up_to_date()
        self._state = self._state.set(force=jax.jit(self._force_fn)(
                            self._state.position,
                            box=self._current_box,
                            nbrs_nm=self._nbrs_nm,
                        ))

    def getVelocity(self):
        '''
            Returns the current velocity (Å/fs) of the atoms.
        '''
        return np.array(self._state.velocity)

    def setVelocity(self, velocity):
        '''
            Set the velocity (Å/fs). Shape (N, 3); optionally (P, N, 3) for PIMD.
        '''
        velocity = jnp.asarray(velocity)
        if self._is_pimd and velocity.ndim == 2:
            velocity = jnp.broadcast_to(velocity, self._state.velocity.shape)
        if self._mobile is not None:
            velocity = velocity * self._mobile[:, None]
        self._state = self._state.set(momentum=self._state.mass * velocity)

    def setRandomVelocity(self, temperature, remove_com_motion=True, seed=None):
        """
            Set velocities from a Maxwell-Boltzmann distribution at a given temperature (K).
        """
        if seed is None:
            seed = np.random.randint(1, 1e6)
        key = jax.random.PRNGKey(seed)
        kT = temperature * TEMP_UNIT_CONVERSION
        velocity = jax.random.normal(key,
                                     shape=self._state.velocity.shape,
                                     dtype=self._state.velocity.dtype)
        velocity *= jnp.sqrt(kT * self._n_bead / self._state.mass)
        self.setVelocity(velocity)
        if remove_com_motion:
            self.remove_com_motion()

    def getBox(self):
        '''
            Returns the current box size (Å).
        '''
        return np.array(self._current_box)

    def getEnergy(self):
        '''
            Returns the energy (eV) of the current state.
            Spring term excluded under PIMD.
        '''
        return jax.jit(self._energy_fn)(
                            self._state.position,
                            box=self._current_box,
                            nbrs_nm=self._nbrs_nm,
                        ).item()

    def getForce(self):
        '''
            Returns the force (eV/Å) of the current state, shape (N, 3) or
            (P, N, 3) under PIMD. Spring term excluded under PIMD.
        '''
        if not self._is_pimd:
            return np.array(self._state.force)
        # _energy_fn is bead-averaged; multiply by n_bead so each bead's
        # gradient yields its own ML force.
        f = -jax.jit(jax.grad(lambda r, **kw: self._n_bead * self._energy_fn(r, **kw)))(
            self._state.position,
            box=self._current_box,
            nbrs_nm=self._nbrs_nm,
        )
        return np.array(f)

    def getPressure(self):
        '''
            Returns the pressure (bar) of the current state.
        '''
        return jax.jit(self._pressure_fn)(
                                self._state,
                                self._current_box,
                                self._nbrs_nm,
                            ).item()





# Back-compat re-exports; these classes have moved to deprecated.py.
# Imported at the bottom to avoid circular import (deprecated.py imports Simulation).
from .deprecated import TrajDump, TrajDumpSimulation  # noqa: E402, F401
