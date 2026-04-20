# overrides certain jax_md routines

import jax_md
import jax
import jax.numpy as jnp
from jax_md import quantity, util
from jax_md import simulate as _jms

def nvt_nose_hoover_invariant(energy_fn,
                    state: jax_md.simulate.NVTNoseHooverState,
                    kT: float,
                    **kwargs) -> float:
    """The conserved quantity for the NVT ensemble with a Nose-Hoover thermostat.

    This function is adapted from jax_md.simulate to define the DOF differently.

    Arguments:
        energy_fn: The energy function of the Nose-Hoover system.
        state: The current state of the system.
        kT: The current goal temperature of the system.

    Returns:
        The Hamiltonian of the extended NVT dynamics.
    """
    PE = energy_fn(state.position, **kwargs)
    KE = jax_md.simulate.kinetic_energy(state)

    DOF = state.chain.degrees_of_freedom
    E = PE + KE

    c = state.chain

    E += c.momentum[0] ** 2 / (2 * c.mass[0]) + DOF * kT * c.position[0]
    for r, p, m in zip(c.position[1:], c.momentum[1:], c.mass[1:]):
        E += p ** 2 / (2 * m) + kT * r
    return E

def nve_with_fixed_atoms(nve_init_fn, nve_apply_fn, mobile_mask):
    """Wrap NVE init/step functions to handle fixed atoms correctly."""
    mobile = mobile_mask[:, None].astype(jnp.float32)  # (N, 1)

    def init_fn(key, R, *args, **kwargs):
        state = nve_init_fn(key, R, *args, **kwargs)
        return state.set(momentum=state.momentum * mobile)

    def apply_fn(state, *args, **kwargs):
        state = nve_apply_fn(state, *args, **kwargs)
        return state.set(momentum=state.momentum * mobile)

    return init_fn, apply_fn

def nvt_with_fixed_atoms(nvt_init_fn, nvt_apply_fn, mobile_mask):
    """Wrap NVT init/step functions to handle fixed atoms correctly."""
    mobile = mobile_mask[:, None].astype(jnp.float32)  # (N, 1)
    dof = int(3 * jnp.sum(mobile_mask))

    def init_fn(key, R, *args, **kwargs):
        state = nvt_init_fn(key, R, *args, **kwargs)
        # Zero fixed atom velocities
        state = state.set(momentum=state.momentum * mobile)
        # Fix thermostat chain: correct DOF, chain mass Q[0], and stored KE
        if hasattr(state, "chain"):
            KE = jax_md.simulate.kinetic_energy(state)
            new_mass = state.chain.mass.at[0].multiply(dof / R.size)
            state = state.set(chain=state.chain.set(
                degrees_of_freedom=dof,
                kinetic_energy=KE,
                mass=new_mass,
            ))
        return state

    def apply_fn(state, *args, **kwargs):
        # Ensure thermostat DOF is correct before step
        if hasattr(state, "chain"):
            state = state.set(chain=state.chain.set(degrees_of_freedom=dof))

        state = nvt_apply_fn(state, *args, **kwargs)

        # Zero fixed momenta after thermostat update
        state = state.set(momentum=state.momentum * mobile)
        return state

    return init_fn, apply_fn


@jax_md.dataclasses.dataclass
class CachedNPTNoseHooverState:
    """Mirrors jax_md.simulate.NPTNoseHooverState with an extra cached dU/dV.

    Storing dUdV lets inner_step skip the initial stress_fn call and perform a
    single combined value_and_grad per step (see jax-md issue #330).
    """
    position: jnp.ndarray
    momentum: jnp.ndarray
    force: jnp.ndarray
    mass: jnp.ndarray

    reference_box: jnp.ndarray

    box_position: jnp.ndarray
    box_momentum: jnp.ndarray
    box_mass: jnp.ndarray

    barostat: _jms.NoseHooverChain
    thermostat: _jms.NoseHooverChain

    dUdV: jnp.ndarray

    @property
    def velocity(self):
        return self.momentum / self.mass

    @property
    def box(self):
        dim = self.position.shape[1]
        ref = self.reference_box
        V_0 = quantity.volume(dim, ref)
        V = V_0 * jnp.exp(dim * self.box_position)
        return (V / V_0) ** (1 / dim) * ref


def npt_nose_hoover(
    energy_fn,
    shift_fn,
    dt,
    pressure,
    kT,
    barostat_kwargs=None,
    thermostat_kwargs=None,
):
    """NPT Nose-Hoover integrator that caches dU/dV across steps.

    Adapted from jax_md.simulate.npt_nose_hoover. Each step needs only one
    combined value_and_grad (force + dU/dV) instead of two; the dU/dV from the
    end of step n is reused as the dU/dV at the start of step n+1.
    """
    f32 = jnp.float32
    dt = f32(dt)
    dt_2 = f32(dt / 2)

    barostat_kwargs = _jms.default_nhc_kwargs(1000 * dt, barostat_kwargs)
    barostat = _jms.nose_hoover_chain(dt, **barostat_kwargs)

    thermostat_kwargs = _jms.default_nhc_kwargs(100 * dt, thermostat_kwargs)
    thermostat = _jms.nose_hoover_chain(dt, **thermostat_kwargs)

    def force_stress_fn(position, box, **kwargs):
        def U(position, eps):
            return energy_fn(position, box=box, perturbation=(1 + eps), **kwargs)
        E, grad = jax.value_and_grad(U, argnums=(0, 1))(position, 0.0)
        return E, -grad[0], grad[1]

    def box_force(alpha, vol, dUdV, momentum, mass, pressure):
        _, dim = momentum.shape
        KE2 = util.high_precision_sum(momentum ** 2 / mass)
        return alpha * KE2 - dUdV - pressure * vol * dim

    def sinhx_x(x):
        return (1 + x ** 2 / 6 + x ** 4 / 120 + x ** 6 / 5040
                + x ** 8 / 362_880 + x ** 10 / 39_916_800)

    def exp_iL1(box, R, V, V_b, **kwargs):
        x = V_b * dt
        x_2 = x / 2
        sinhV = sinhx_x(x_2)
        return shift_fn(R, R * (jnp.exp(x) - 1) + dt * V * jnp.exp(x_2) * sinhV,
                        box=box, **kwargs)

    def exp_iL2(alpha, P, F, V_b):
        x = alpha * V_b * dt_2
        x_2 = x / 2
        sinhP = sinhx_x(x_2)
        return P * jnp.exp(-x) + dt_2 * F * sinhP * jnp.exp(-x_2)

    def _box_info(state):
        dim = state.position.shape[1]
        ref = state.reference_box
        V_0 = quantity.volume(dim, ref)
        V = V_0 * jnp.exp(dim * state.box_position)
        return V, lambda V: (V / V_0) ** (1 / dim) * ref

    def update_box_mass(state, kT):
        N, dim = state.position.shape
        dtype = state.position.dtype
        box_mass = jnp.array(dim * (N + 1) * kT * state.barostat.tau ** 2, dtype)
        return state.set(box_mass=box_mass)

    def init_fn(key, R, box, mass=f32(1.0), **kwargs):
        N, dim = R.shape
        _kT = kwargs.get('kT', kT)

        zero = jnp.zeros((), dtype=R.dtype)
        one = jnp.ones((), dtype=R.dtype)
        box_position = zero
        box_momentum = zero
        box_mass = dim * (N + 1) * _kT * barostat_kwargs['tau'] ** 2 * one
        KE_box = quantity.kinetic_energy(momentum=box_momentum, mass=box_mass)

        if jnp.isscalar(box) or box.ndim == 0:
            box = jnp.eye(R.shape[-1]) * box

        # One combined call seeds both force and dU/dV for the cache.
        _, F, dUdV = force_stress_fn(R, box,
                                     **{k: v for k, v in kwargs.items() if k != 'kT'})

        state = CachedNPTNoseHooverState(
            position=R,
            momentum=None,
            force=F,
            mass=mass,
            reference_box=box,
            box_position=box_position,
            box_momentum=box_momentum,
            box_mass=box_mass,
            barostat=barostat.initialize(1, KE_box, _kT),
            thermostat=None,
            dUdV=dUdV,
        )
        state = _jms.canonicalize_mass(state)
        state = _jms.initialize_momenta(state, key, _kT)
        KE = _jms.kinetic_energy(state)
        return state.set(
            thermostat=thermostat.initialize(quantity.count_dof(R), KE, _kT)
        )

    def inner_step(state, **kwargs):
        _pressure = kwargs.pop('pressure', pressure)

        R, P, M, F = state.position, state.momentum, state.mass, state.force
        R_b, P_b, M_b = state.box_position, state.box_momentum, state.box_mass
        dUdV = state.dUdV

        N, _ = R.shape
        alpha = 1 + 1 / N

        vol, _ = _box_info(state)
        # First barostat kick uses cached dU/dV from previous step (or init).
        G_e = box_force(alpha, vol, dUdV, P, M, _pressure)
        P_b = P_b + dt_2 * G_e
        P = exp_iL2(alpha, P, F, P_b / M_b)

        R_b = R_b + P_b / M_b * dt
        state = state.set(box_position=R_b)

        vol, box_fn = _box_info(state)
        box = box_fn(vol)
        R = exp_iL1(box, R, P / M, P_b / M_b)

        # Single combined gradient call per step.
        _, F, dUdV = force_stress_fn(R, box, **kwargs)

        P = exp_iL2(alpha, P, F, P_b / M_b)
        G_e = box_force(alpha, vol, dUdV, P, M, _pressure)
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