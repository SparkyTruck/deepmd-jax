# overrides certain jax_md routines

import jax_md
import jax
import jax.numpy as jnp
import numpy as np
from jax_md import quantity, util, simulate

def nvt_nose_hoover_invariant(energy_fn,
                    state: jax_md.simulate.NVTNoseHooverState,
                    kT: float,
                    **kwargs) -> float:
    """Adapted from jax_md.simulate to define the DOF differently."""
    PE = energy_fn(state.position, **kwargs)
    KE = jax_md.simulate.kinetic_energy(state)

    DOF = state.chain.degrees_of_freedom
    E = PE + KE

    c = state.chain

    E += c.momentum[0] ** 2 / (2 * c.mass[0]) + DOF * kT * c.position[0]
    for r, p, m in zip(c.position[1:], c.momentum[1:], c.mass[1:]):
        E += p ** 2 / (2 * m) + kT * r
    return E

@jax_md.dataclasses.dataclass
class NVTLangevinStateWithWork:
    """Like ``jax_md.simulate.NVTLangevinState`` plus a cumulative thermostat
    work accumulator. Subtracting ``thermostat_work`` from ``KE + PE`` yields
    the Langevin conserved quantity (analogue of the NVE invariant), exact
    under BAOAB up to BA-step discretization error.
    """
    position: jnp.ndarray
    momentum: jnp.ndarray
    force: jnp.ndarray
    mass: jnp.ndarray
    rng: jnp.ndarray
    thermostat_work: jnp.ndarray

    @property
    def velocity(self):
        return self.momentum / self.mass


def nvt_langevin(energy_or_force_fn, shift_fn, dt, kT, gamma=0.1,
                 mobile_mask=None, **sim_kwargs):
    """BAOAB Langevin integrator that mirrors ``jax_md.simulate.nvt_langevin``.

    Differences from the upstream version:
      * the state carries ``thermostat_work``, the cumulative work done by the
        O step (Σ KE_after_O − KE_before_O); subtract it from KE + PE to get
        the Langevin invariant.
      * ``mobile_mask`` (optional, shape ``(N,)``): atoms with mask=False keep
        zero momentum throughout — both at init and during each O step — so
        they never enter ``thermostat_work``. Forces on those atoms are
        expected to be zero already (use ``stop_gradient`` in the energy fn).
    """
    force_fn = quantity.canonicalize_force(energy_or_force_fn)
    if mobile_mask is not None:
        mobile = jnp.asarray(mobile_mask, dtype=jnp.float32)[:, None]
    else:
        mobile = None

    @jax.jit
    def init_fn(key, R, mass=jnp.float32(1.0), **kwargs):
        _kT = kwargs.pop('kT', kT)
        key, split = jax.random.split(key)
        force = force_fn(R, **kwargs)
        state = NVTLangevinStateWithWork(
            position=R, momentum=None, force=force, mass=mass, rng=key,
            thermostat_work=jnp.zeros((), dtype=R.dtype),
        )
        state = simulate.canonicalize_mass(state)
        state = simulate.initialize_momenta(state, split, _kT)
        if mobile is not None:
            state = state.set(momentum=state.momentum * mobile)
        return state

    @jax.jit
    def step_fn(state, **kwargs):
        _dt = kwargs.pop('dt', dt)
        _kT = kwargs.pop('kT', kT)
        dt_2 = _dt / 2

        state = simulate.momentum_step(state, dt_2)
        state = simulate.position_step(state, shift_fn, dt_2, **kwargs)
        # O step (inlined so we can record ΔKE and mask noise on fixed atoms).
        P_old = state.momentum
        c1 = jnp.exp(-gamma * _dt)
        c2 = jnp.sqrt(jnp.maximum(_kT * (1.0 - c1 ** 2), 0.0))
        key, split = jax.random.split(state.rng)
        noise = jax.random.normal(split, shape=P_old.shape, dtype=P_old.dtype)
        if mobile is not None:
            noise = noise * mobile
        P_new = c1 * P_old + c2 * jnp.sqrt(state.mass) * noise
        delta_KE = 0.5 * jnp.sum((P_new ** 2 - P_old ** 2) / state.mass)
        state = state.set(
            momentum=P_new, rng=key,
            thermostat_work=state.thermostat_work + delta_KE,
        )
        state = simulate.position_step(state, shift_fn, dt_2, **kwargs)
        state = state.set(force=force_fn(state.position, **kwargs))
        state = simulate.momentum_step(state, dt_2)
        return state

    return init_fn, step_fn


def with_fixed_atoms(init_fn, apply_fn, mobile_mask):
    """Wrap NVE / Nose-Hoover NVT to constrain fixed atoms. Masks momenta;
    fixes Nose-Hoover chain DOF / mass / KE if the state has a ``chain``.
    Not suitable for Langevin (use the integrator's own ``mobile_mask``)."""
    mobile = mobile_mask[:, None].astype(jnp.float32)  # (N, 1)
    dof = int(3 * jnp.sum(mobile_mask))

    def wrapped_init(key, R, *args, **kwargs):
        state = init_fn(key, R, *args, **kwargs)
        state = state.set(momentum=state.momentum * mobile)
        if hasattr(state, "chain"):
            KE = jax_md.simulate.kinetic_energy(state)
            new_mass = state.chain.mass.at[0].multiply(dof / R.size)
            state = state.set(chain=state.chain.set(
                degrees_of_freedom=dof,
                kinetic_energy=KE,
                mass=new_mass,
            ))
        return state

    def wrapped_apply(state, *args, **kwargs):
        if hasattr(state, "chain"):
            state = state.set(chain=state.chain.set(degrees_of_freedom=dof))
        state = apply_fn(state, *args, **kwargs)
        return state.set(momentum=state.momentum * mobile)

    return wrapped_init, wrapped_apply


def nve(energy_or_force_fn, shift_fn, dt, mobile_mask=None, **kwargs):
    """jax_md NVE integrator with optional fixed-atom support."""
    init_fn, apply_fn = simulate.nve(energy_or_force_fn, shift_fn, dt, **kwargs)
    if mobile_mask is None:
        return init_fn, apply_fn
    return with_fixed_atoms(init_fn, apply_fn, mobile_mask)


def nvt_nose_hoover(energy_or_force_fn, shift_fn, dt, kT, tau=None,
                    chain_length=3, chain_steps=1, sy_steps=1,
                    mobile_mask=None):
    """jax_md NVT Nose-Hoover integrator with optional fixed-atom support."""
    init_fn, apply_fn = simulate.nvt_nose_hoover(
        energy_or_force_fn, shift_fn, dt, kT,
        tau=tau, chain_length=chain_length,
        chain_steps=chain_steps, sy_steps=sy_steps,
    )
    if mobile_mask is None:
        return init_fn, apply_fn
    return with_fixed_atoms(init_fn, apply_fn, mobile_mask)


@jax_md.dataclasses.dataclass
class CachedNPTNoseHooverState:
    """Mirrors jax_md.simulate.NPTNoseHooverState with an extra cached dUdV.
    Supports orthorhombic anisotropic NPT where axes are partitioned into G
    coupled groups (each sharing a single scale factor) plus optionally fixed
    axes. group_ids is a (3,) int array: entry a is the group index of axis a,
    or -1 if axis a is fixed. box_position / box_momentum / box_mass are (G,).
    group_ids = [0,0,0], G=1 recovers standard isotropic NPT.
    """
    position: jnp.ndarray
    momentum: jnp.ndarray
    force: jnp.ndarray
    mass: jnp.ndarray

    reference_box: jnp.ndarray

    box_position: jnp.ndarray
    box_momentum: jnp.ndarray
    box_mass: jnp.ndarray

    barostat: simulate.NoseHooverChain
    thermostat: simulate.NoseHooverChain

    dUdV: jnp.ndarray
    group_ids: jnp.ndarray

    @property
    def velocity(self):
        return self.momentum / self.mass

    @property
    def box(self):
        scales_g = jnp.exp(self.box_position)
        safe = jnp.maximum(self.group_ids, 0)
        per_axis = jnp.where(self.group_ids >= 0, scales_g[safe], 1.)
        return self.reference_box * per_axis


def _parse_couple_axes(couple_axes):
    """Validate couple_axes and return (n_per_group, group_ids_np, membership_np).
    couple_axes is a tuple-of-tuples partitioning a subset of {0,1,2} into
    coupled groups. Any axis not listed is held fixed.
    """
    try:
        groups = tuple(tuple(int(a) for a in g) for g in couple_axes)
    except TypeError:
        raise ValueError(
            f"couple_axes must be a tuple of tuples of ints; got {couple_axes}"
        )
    all_axes = [a for g in groups for a in g]
    if any(a not in (0, 1, 2) for a in all_axes):
        raise ValueError(f"couple_axes axes must be in {{0,1,2}}; got {couple_axes}")
    if len(set(all_axes)) != len(all_axes):
        raise ValueError(f"couple_axes must be a partition (no repeats); got {couple_axes}")
    if any(len(g) == 0 for g in groups):
        raise ValueError(f"couple_axes groups must be non-empty; got {couple_axes}")
    if len(groups) == 0:
        raise ValueError("couple_axes must contain at least one group; use NVT instead")
    G = len(groups)
    n_per_group = np.array([len(g) for g in groups], dtype=np.int32)
    group_ids_np = np.full(3, -1, dtype=np.int32)
    membership_np = np.zeros((3, G), dtype=np.float32)
    for g_idx, group in enumerate(groups):
        for a in group:
            group_ids_np[a] = g_idx
            membership_np[a, g_idx] = 1.0
    return groups, n_per_group, group_ids_np, membership_np


def npt_nose_hoover(
    energy_fn,
    shift_fn,
    dt,
    pressure,
    kT,
    barostat_kwargs=None,
    thermostat_kwargs=None,
    couple_axes=((0, 1, 2),),
):
    """Adapted from jax_md.simulate.npt_nose_hoover. dUdV from the
    end of step n is reused as the dU/dV at the start of step n+1.
    Positions are real-space (Å). 
    couple_axes: partition of a subset of {0,1,2} into coupled groups. Each
    inner tuple lists axes sharing one scale factor. Axes not listed are fixed.
    Default ((0,1,2),) is isotropic. Other examples: ((0,1),) semi-iso (xy
    couple, z fixed); ((0,),) uniaxial (x moves, yz fixed); ((0,),(1,),(2,))
    fully anisotropic ortho; ((0,1),(2,)) xy couple + z independent.
    """
    f32 = jnp.float32
    dt = f32(dt)
    dt_2 = f32(dt / 2)

    _, n_per_group_np, group_ids_np, membership_np = _parse_couple_axes(couple_axes)
    G = n_per_group_np.shape[0]
    group_ids = jnp.asarray(group_ids_np)
    membership = jnp.asarray(membership_np)  # (3, G), per-axis one-hot over groups
    n_per_group = jnp.asarray(n_per_group_np, dtype=f32)  # (G,)

    barostat_kwargs = simulate.default_nhc_kwargs(1000 * dt, barostat_kwargs)
    barostat = simulate.nose_hoover_chain(dt, **barostat_kwargs)

    thermostat_kwargs = simulate.default_nhc_kwargs(100 * dt, thermostat_kwargs)
    thermostat = simulate.nose_hoover_chain(dt, **thermostat_kwargs)

    def force_stress_fn(position, box, **kwargs):
        def U(position, eps):
            # eps is (G,); perturbation[a] = 1 + membership[a,:] . eps.
            # Fixed axes have membership[a,:]=0, so perturbation stays 1.
            perturbation = 1.0 + membership @ eps
            return energy_fn(position, box=box, perturbation=perturbation, **kwargs)
        eps0 = jnp.zeros(G, dtype=position.dtype)
        E, grad = jax.value_and_grad(U, argnums=(0, 1))(position, eps0)
        return E, -grad[0], grad[1]  # grad[1]: (G,), per-group dU/deps

    def _per_axis_rate(V_b, N_f):
        """Per-axis momentum rescale rate: V_b[g(a)] + tr(V_b)/N_f.
        Fixed axes get the bath-coupling term tr(V_b)/N_f only (MTTK correction).
        """
        V_b_per_axis = membership @ V_b  # (3,) zero on fixed axes
        tr_Vb = jnp.sum(n_per_group * V_b)
        return V_b_per_axis + tr_Vb / N_f

    def box_force(V, dUdV, momentum, mass, pressure, N_f):
        # Per-axis KE2 along each axis, summed over atoms.
        KE2_axis = util.high_precision_sum(momentum ** 2 / mass, axis=0)  # (3,)
        KE2_group = membership.T @ KE2_axis  # (G,)
        KE2_total = jnp.sum(KE2_axis)
        # G_g = KE2_axial_g + (n_g/N_f) * KE2_total - dUdV_g - n_g * V * P
        return (KE2_group + (n_per_group / N_f) * KE2_total
                - dUdV - pressure * V * n_per_group)

    def sinhx_x(x):
        return (1 + x ** 2 / 6 + x ** 4 / 120 + x ** 6 / 5040
                + x ** 8 / 362_880 + x ** 10 / 39_916_800)

    def exp_iL1(box, R, V, V_b, **kwargs):
        # Position scales only along axes in moving groups.
        V_b_per_axis = membership @ V_b  # (3,) zero on fixed axes
        x = V_b_per_axis * dt
        x_2 = x / 2
        sinhV = sinhx_x(x_2)
        return shift_fn(R, R * (jnp.exp(x) - 1) + dt * V * jnp.exp(x_2) * sinhV,
                        box=box, **kwargs)

    def exp_iL2(P, F, V_b, N_f):
        rate = _per_axis_rate(V_b, N_f)  # (3,)
        x = rate * dt_2
        x_2 = x / 2
        sinhP = sinhx_x(x_2)
        return P * jnp.exp(-x) + dt_2 * F * sinhP * jnp.exp(-x_2)

    def _box_and_vol(state):
        ref = state.reference_box
        dim = state.position.shape[1]
        V_0 = quantity.volume(dim, ref)
        scales_g = jnp.exp(state.box_position)  # (G,)
        per_axis = membership @ scales_g + (1.0 - membership.sum(axis=1))  # (3,)
        box = ref * per_axis
        # V = V_0 * prod_g scales_g^{n_g}; fixed axes contribute factor 1.
        V = V_0 * jnp.prod(per_axis)
        return box, V

    def update_box_mass(state, kT):
        N, _ = state.position.shape
        dtype = state.position.dtype
        box_mass = jnp.asarray(n_per_group * (N + 1) * kT * state.barostat.tau ** 2,
                                dtype)
        return state.set(box_mass=box_mass)

    def init_fn(key, R, box, mass=f32(1.0), **kwargs):
        N, dim = R.shape
        _kT = kwargs.get('kT', kT)

        box_position = jnp.zeros(G, dtype=R.dtype)
        box_momentum = jnp.zeros(G, dtype=R.dtype)
        box_mass = jnp.asarray(n_per_group * (N + 1) * _kT * barostat_kwargs['tau'] ** 2,
                                dtype=R.dtype)
        KE_box = quantity.kinetic_energy(momentum=box_momentum, mass=box_mass)

        if jnp.isscalar(box) or box.ndim == 0:
            box = jnp.eye(R.shape[-1]) * box

        _, F, dUdV = force_stress_fn(R, box, **{k: v for k, v in kwargs.items() if k != 'kT'})

        state = CachedNPTNoseHooverState(
            position=R,
            momentum=None,
            force=F,
            mass=mass,
            reference_box=box,
            box_position=box_position,
            box_momentum=box_momentum,
            box_mass=box_mass,
            barostat=barostat.initialize(G, KE_box, _kT),
            thermostat=None,
            dUdV=dUdV,
            group_ids=group_ids,
        )
        state = simulate.canonicalize_mass(state)
        state = simulate.initialize_momenta(state, key, _kT)
        KE = simulate.kinetic_energy(state)
        return state.set(
            thermostat=thermostat.initialize(quantity.count_dof(R), KE, _kT)
        )

    def inner_step(state, **kwargs):
        _pressure = kwargs.pop('pressure', pressure)

        R, P, M, F = state.position, state.momentum, state.mass, state.force
        R_b, P_b, M_b = state.box_position, state.box_momentum, state.box_mass
        dUdV = state.dUdV

        N, dim = R.shape
        N_f = f32(dim * N)  # total translational DOF (CoM subtraction negligible)

        _, vol = _box_and_vol(state)
        # First barostat kick uses cached dU/dV from previous step (or init).
        G_e = box_force(vol, dUdV, P, M, _pressure, N_f)
        P_b = P_b + dt_2 * G_e
        P = exp_iL2(P, F, P_b / M_b, N_f)

        R_b = R_b + P_b / M_b * dt
        state = state.set(box_position=R_b)

        box, vol = _box_and_vol(state)
        R = exp_iL1(box, R, P / M, P_b / M_b)

        _, F, dUdV = force_stress_fn(R, box, **kwargs)

        P = exp_iL2(P, F, P_b / M_b, N_f)
        G_e = box_force(vol, dUdV, P, M, _pressure, N_f)
        P_b = P_b + dt_2 * G_e

        return state.set(
            position=R, momentum=P, mass=M, force=F,
            box_position=R_b, box_momentum=P_b, box_mass=M_b,
            dUdV=dUdV,
        )

    def apply_fn(state, **kwargs):
        S = state
        _kT = kwargs.get('kT', kT)

        bc = barostat.update_mass(S.barostat, _kT)
        tc = thermostat.update_mass(S.thermostat, _kT)
        S = update_box_mass(S, _kT)

        P_b, bc = barostat.half_step(S.box_momentum, bc, _kT)
        P, tc = thermostat.half_step(S.momentum, tc, _kT)

        S = S.set(momentum=P, box_momentum=P_b)
        S = inner_step(S, **kwargs)

        KE = quantity.kinetic_energy(momentum=S.momentum, mass=S.mass)
        tc = tc.set(kinetic_energy=KE)

        KE_box = quantity.kinetic_energy(momentum=S.box_momentum, mass=S.box_mass)
        bc = bc.set(kinetic_energy=KE_box)

        P, tc = thermostat.half_step(S.momentum, tc, _kT)
        P_b, bc = barostat.half_step(S.box_momentum, bc, _kT)

        S = S.set(thermostat=tc, barostat=bc, momentum=P, box_momentum=P_b)
        return S

    return init_fn, apply_fn


def npt_nose_hoover_invariant(energy_fn, state, pressure, kT, **kwargs):
    """Conserved quantity for CachedNPTNoseHooverState (multi-group NPT)."""
    ref = state.reference_box
    dim = state.position.shape[1]
    V_0 = quantity.volume(dim, ref)
    scales_g = jnp.exp(state.box_position)  # (G,)
    safe = jnp.maximum(state.group_ids, 0)
    per_axis = jnp.where(state.group_ids >= 0, scales_g[safe], 1.)
    box = ref * per_axis
    V = V_0 * jnp.prod(per_axis)

    PE = energy_fn(state.position, box=box, **kwargs)
    KE = jax_md.simulate.kinetic_energy(state)
    DOF = state.position.size
    E = PE + KE

    c = state.thermostat
    E += c.momentum[0] ** 2 / (2 * c.mass[0]) + DOF * kT * c.position[0]
    for r, p, m in zip(c.position[1:], c.momentum[1:], c.mass[1:]):
        E += p ** 2 / (2 * m) + kT * r

    c = state.barostat
    for r, p, m in zip(c.position, c.momentum, c.mass):
        E += p ** 2 / (2 * m) + kT * r

    E += pressure * V
    E += jnp.sum(state.box_momentum ** 2 / (2 * state.box_mass))
    return E
