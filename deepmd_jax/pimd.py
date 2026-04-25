"""Path-integral (ring-polymer) molecular dynamics support.

Provides a BAOAB Langevin integrator that mirrors ``jax_md.simulate.nvt_langevin``
but operates on ring-polymer states shaped ``(P, N, 3)`` and replaces the O step
with the PILE-L thermostat (Langevin on normal-mode momenta).

Spring couplings between beads are carried by the user-supplied energy function
so ``force = -grad(V_ML/P + V_spring)`` flows through the B step exactly as in the
classical Langevin case. This keeps the integrator identical in shape to jax_md's
reference implementation; the price is a dt limit of roughly 1/omega_max where
omega_max ~ 2 * n_bead * k_B T / hbar.
"""

import jax
import jax.numpy as jnp
import numpy as np
import jax_md
from jax_md import simulate as _jms
from jax_md import quantity

HBAR = 6.582119569e-16 * 1e15  # reduced Planck's constant, eV*fs


def normal_mode_transform(n_bead, kT_phys):
    """Normal-mode frequencies and orthogonal transform for a ring polymer.

    Returns
    -------
    freqs : np.ndarray, shape ``(P,)``
        Normal-mode angular frequencies in fs^-1. ``freqs[0]`` is 0 (centroid).
    C : np.ndarray, shape ``(P, P)``
        Real orthogonal matrix mapping normal-mode to primitive coordinates:
        ``R_primitive = C @ R_nm``; inversely ``R_nm = C.T @ R_primitive``.
    """
    ring_poly_freq = n_bead * kT_phys / HBAR
    A = (2 * np.eye(n_bead)
         - np.roll(np.eye(n_bead), 1, axis=0)
         - np.roll(np.eye(n_bead), -1, axis=0))
    eva, eve = np.linalg.eigh(A)
    order = np.argsort(eva)
    eva = eva[order]
    eve = eve[:, order]
    if n_bead >= 2:
        eva[0] = 0.0  # clean numerical noise on the centroid eigenvalue
    freqs = ring_poly_freq * np.sqrt(np.maximum(eva, 0.0))
    return freqs, eve


@jax_md.dataclasses.dataclass
class PIMDLangevinState:
    """State for BAOAB Langevin PIMD with PILE-L thermostat.

    Attributes
    ----------
    position, momentum, force : ``(P, N, 3)`` arrays.
    mass : ``(N, 1)`` array (canonicalized from a ``(N,)`` input).
    rng : PRNG key.
    """
    position: jnp.ndarray
    momentum: jnp.ndarray
    force: jnp.ndarray
    mass: jnp.ndarray
    rng: jnp.ndarray

    @property
    def velocity(self):
        return self.momentum / self.mass


@_jms.initialize_momenta.register(PIMDLangevinState)
def _initialize_momenta_pimd(state, key, kT):
    """Maxwell-Boltzmann at the *effective* temperature (``kT_eff = P * kT_phys``).

    We deliberately do **not** subtract the mean along the bead axis (that would
    zero the centroid momentum). The user can explicitly remove physical CoM
    motion on the centroid via ``Simulation.remove_com_motion`` if desired.
    """
    shape = state.position.shape  # (P, N, 3)
    p = jnp.sqrt(state.mass * kT) * jax.random.normal(key, shape, dtype=state.position.dtype)
    return state.set(momentum=p)


def _pile_l_stochastic_step(state, dt, kT_eff, gamma_P, nm_trans):
    """O step in the normal-mode basis.

    gamma_P : ``(P,)`` per-mode friction.
    nm_trans : ``(P, P)`` orthogonal transform ``C``.
    """
    # Transform primitive momenta -> normal-mode momenta via C.T.
    P_nm = jnp.tensordot(nm_trans.T, state.momentum, axes=(1, 0))  # (P, N, 3)
    c1 = jnp.exp(-gamma_P * dt)[:, None, None]                       # (P, 1, 1)
    c2 = jnp.sqrt(jnp.maximum(kT_eff * (1.0 - c1**2), 0.0))           # (P, 1, 1)
    key, split = jax.random.split(state.rng)
    noise = jax.random.normal(split, shape=P_nm.shape, dtype=P_nm.dtype)
    # Variance m*kT is preserved by the orthogonal transform; sampling per-mode
    # at c2^2 * m matches the primitive Langevin reference in the centroid mode
    # and critically damps the internal modes.
    P_nm = c1 * P_nm + c2 * jnp.sqrt(state.mass) * noise
    momentum = jnp.tensordot(nm_trans, P_nm, axes=(1, 0))            # back to primitive
    return state.set(momentum=momentum, rng=key)


def nvt_langevin_pimd(
    energy_or_force_fn,
    shift_fn,
    dt,
    kT_eff,
    gamma_P,
    nm_trans,
    **sim_kwargs,
):
    """BAOAB Langevin integrator for PIMD with a PILE-L thermostat.

    Mirrors ``jax_md.simulate.nvt_langevin``; only the O (stochastic) step
    differs. Spring couplings must be contained inside ``energy_or_force_fn``.

    Args
    ----
    energy_or_force_fn : callable
        ``energy_fn(R, **kwargs) -> scalar`` with ``R`` of shape ``(P, N, 3)``.
        Must include both the ML potential (averaged over beads) and the
        ring-polymer spring term.
    shift_fn : callable
        Shift applied by ``position_step``; must broadcast over a leading bead
        axis (jax_md's ``periodic`` / ``periodic_general`` do).
    dt : float, time step in fs.
    kT_eff : float, effective temperature ``P * k_B T`` (eV).
    gamma_P : jnp.ndarray, shape ``(P,)``
        Per-normal-mode friction coefficient in fs^-1.
    nm_trans : jnp.ndarray, shape ``(P, P)``
        Orthogonal transform ``C`` (primitive <- normal-mode).
    """
    force_fn = jax_md.quantity.canonicalize_force(energy_or_force_fn)

    @jax.jit
    def init_fn(key, R, mass=jnp.float32(1.0), **kwargs):
        _kT = kwargs.pop('kT', kT_eff)
        key, split = jax.random.split(key)
        force = force_fn(R, **kwargs)
        state = PIMDLangevinState(
            position=R, momentum=None, force=force, mass=mass, rng=key,
        )
        state = _jms.canonicalize_mass(state)
        return _jms.initialize_momenta(state, split, _kT)

    @jax.jit
    def step_fn(state, **kwargs):
        _dt = kwargs.pop('dt', dt)
        _kT = kwargs.pop('kT', kT_eff)
        dt_2 = _dt / 2

        state = _jms.momentum_step(state, dt_2)
        state = _jms.position_step(state, shift_fn, dt_2, **kwargs)
        state = _pile_l_stochastic_step(state, _dt, _kT, gamma_P, nm_trans)
        state = _jms.position_step(state, shift_fn, dt_2, **kwargs)
        state = state.set(force=force_fn(state.position, **kwargs))
        state = _jms.momentum_step(state, dt_2)
        return state

    return init_fn, step_fn


def unwrap_across_beads(coord, box):
    """Minimum-image-shift every bead onto the same periodic image as bead 0.

    Without this, jax_md's per-bead shift_fn can place beads k and k-1 on
    opposite sides of the periodic boundary, which makes the ring-polymer spring
    term ``(R_k - R_{k-1})**2`` evaluate to ``O(box)**2`` instead of the tiny
    thermal-de-Broglie-scale separation. It also breaks naive ``.mean(axis=0)``
    centroid reductions.

    Parameters
    ----------
    coord : jnp.ndarray, shape ``(P, N, 3)``
        Per-bead positions, possibly wrapped independently.
    box : jnp.ndarray, shape ``(3,)`` or ``(3, 3)``
        The simulation cell. Must be shared across beads.

    Returns
    -------
    coord_consistent : jnp.ndarray, shape ``(P, N, 3)``
        Bead 0 unchanged; beads 1..P-1 shifted to bead 0's image by the
        minimum-image convention. Useful upstream of spring-energy evaluation
        and any centroid reduction.
    """
    ref = coord[0:1]                          # (1, N, 3)
    delta = coord - ref
    if box.size == 3:                         # orthorhombic (3,)
        box_vec = box.reshape(3)
        delta = delta - box_vec * jnp.round(delta / box_vec)
    else:                                     # general (3, 3)
        inv_box = jnp.linalg.inv(box)
        frac = delta @ inv_box
        frac = frac - jnp.round(frac)
        delta = frac @ box
    return ref + delta


def centroid(coord, box):
    """Ring-polymer centroid with correct handling of periodic wrapping.

    Equivalent to ``unwrap_across_beads(coord, box).mean(axis=0)`` — included as
    a top-level helper so users can lift a trajectory back to centroid space
    without reimplementing the unwrap.
    """
    return unwrap_across_beads(coord, box).mean(axis=0)


def primitive_kinetic_energy(state):
    """Kinetic energy of the extended ring-polymer system (all P*N primitive DOFs)."""
    return quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)


def physical_temperature(state, n_bead):
    """Physical temperature (in the same units as ``quantity.temperature``).

    The primitive velocities sample at ``kT_eff = P * kT_phys``; divide by P.
    """
    return quantity.temperature(momentum=state.momentum, mass=state.mass) / n_bead
