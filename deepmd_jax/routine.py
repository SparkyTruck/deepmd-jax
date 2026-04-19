# overrides certain jax_md routines

import jax_md
import jax
import jax.numpy as jnp

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