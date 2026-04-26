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


def cv_pressure_from_dUdV_iso(N, dUdV_iso, V, kT_phys):
    """Centroid-virial pressure (eV/Å^3) from the isotropic ML strain derivative.

    P_cv = (N kT - (1/3) dU_ml/d_eps_iso) / V.

    dU_ml/d_eps is the bead-averaged ML energy's derivative under uniform strain
    eps -> (1+eps) on both R and box. PBC is handled inside the energy_fn (via
    box scaling), so this is the correct estimator under periodic boundaries.
    """
    return (N * kT_phys - dUdV_iso / 3.0) / V


# -----------------------------------------------------------------------------
# NPT-PIMD: Parrinello-Rahman on the centroid mode + Langevin barostat.
# -----------------------------------------------------------------------------


@jax_md.dataclasses.dataclass
class PIMDNPTLangevinState:
    """State for BAOAB Langevin PIMD with PILE-L thermostat and a Langevin barostat.

    Position/momentum/force are ``(P, N, 3)``; box DOF is ``(G,)`` per couple-axes
    group. ``dUdV`` is the ML-only ``dU/d_eps`` (centroid-virial); the spring
    term is V-independent in real coords and never enters the barostat.
    """
    position: jnp.ndarray
    momentum: jnp.ndarray
    force: jnp.ndarray
    mass: jnp.ndarray

    reference_box: jnp.ndarray

    box_position: jnp.ndarray
    box_momentum: jnp.ndarray
    box_mass: jnp.ndarray

    dUdV: jnp.ndarray
    group_ids: jnp.ndarray
    rng: jnp.ndarray

    @property
    def velocity(self):
        return self.momentum / self.mass

    @property
    def box(self):
        scales_g = jnp.exp(self.box_position)
        safe = jnp.maximum(self.group_ids, 0)
        per_axis = jnp.where(self.group_ids >= 0, scales_g[safe], 1.)
        return self.reference_box * per_axis


@_jms.initialize_momenta.register(PIMDNPTLangevinState)
def _initialize_momenta_pimd_npt(state, key, kT):
    """Maxwell-Boltzmann at the *effective* temperature (kT_eff = P * kT_phys)."""
    p = jnp.sqrt(state.mass * kT) * jax.random.normal(
        key, state.position.shape, dtype=state.position.dtype)
    return state.set(momentum=p)


def _sinhx_x(x):
    return (1 + x ** 2 / 6 + x ** 4 / 120 + x ** 6 / 5040
            + x ** 8 / 362_880 + x ** 10 / 39_916_800)


def npt_langevin_pimd(
    energy_fn,
    shift_fn,
    dt,
    *,
    ml_energy_fn,
    pressure,
    kT_phys,
    kT_eff,
    gamma_P,
    gamma_box,
    nm_trans,
    membership,
    n_per_group,
    group_ids,
    tau_p,
):
    """BAOAB Langevin integrator for NPT path-integral MD.

    Centroid mode follows Parrinello-Rahman; internal modes drift freely. The
    Langevin barostat couples only to the centroid (its ideal-gas drive uses
    physical kT, not the effective kT_eff that the bead momenta sample at).

    Two energy functions:
        energy_fn    - ML + spring (full integrator energy, used for force).
        ml_energy_fn - bead-averaged ML potential (its dU/d_eps is the
                       centroid-virial virial that drives the barostat).
    The spring term is V-independent in real coords and never enters dU/dV.
    """
    f32 = jnp.float32
    dt = f32(dt)
    dt_2 = f32(dt / 2)
    G = n_per_group.shape[0]
    membership = jnp.asarray(membership)
    n_per_group = jnp.asarray(n_per_group, dtype=f32)
    group_ids = jnp.asarray(group_ids)

    def ml_force_stress_fn(R, box, **kwargs):
        """Returns (E_ml_avg, F_ml_per_bead, dU_ml_avg/d_eps_per_group)."""
        n_bead = R.shape[0]
        def U(R_, eps):
            perturbation = 1.0 + membership @ eps
            # Multiply by n_bead before differentiating so grad(R) gives
            # natural per-bead ML forces (-grad V_ML(R_k)).
            return n_bead * ml_energy_fn(R_, box=box, perturbation=perturbation, **kwargs)
        eps0 = jnp.zeros(G, dtype=R.dtype)
        E, grads = jax.value_and_grad(U, argnums=(0, 1))(R, eps0)
        return E / n_bead, -grads[0], grads[1] / n_bead

    full_force_fn = jax_md.quantity.canonicalize_force(energy_fn)

    def force_full_and_dUdV(R, box, **kwargs):
        """Bead force = -grad(ML + spring); dUdV = ML-only (centroid-virial)."""
        _, _, dUdV_ml = ml_force_stress_fn(R, box, **kwargs)
        F_full = full_force_fn(R, box=box, **kwargs)
        # Energy reported is informational; use ML + spring for completeness.
        E_full = energy_fn(R, box=box, **kwargs)
        return E_full, F_full, dUdV_ml

    def _box_and_vol(R_b, ref_box):
        scales_g = jnp.exp(R_b)
        per_axis = membership @ scales_g + (1.0 - membership.sum(axis=1))
        box = ref_box * per_axis
        V = jnp.linalg.det(box) if box.ndim == 2 else jnp.prod(box)
        return box, V

    def _per_axis_rate(V_b, N_f):
        V_b_per_axis = membership @ V_b
        tr_Vb = jnp.sum(n_per_group * V_b)
        return V_b_per_axis + tr_Vb / N_f

    def box_force_cv(V, dUdV, P_centroid, mass, P_target, N_f):
        """Box-momentum driver. KE2 uses centroid momentum so the equilibrium
        ideal-gas drive is N kT_phys (not N P_bead kT_eff)."""
        KE2_axis = jnp.sum(P_centroid ** 2 / mass, axis=0)            # (3,)
        KE2_group = membership.T @ KE2_axis                            # (G,)
        KE2_total = jnp.sum(KE2_axis)
        return (KE2_group + (n_per_group / N_f) * KE2_total
                - dUdV - P_target * V * n_per_group)

    def exp_iL2_pimd(P, F, V_b, N_f):
        """Half-step momentum update with PR rescale on centroid mode only."""
        rate = _per_axis_rate(V_b, N_f)
        x = rate * dt_2
        x_2 = x / 2
        sinhP = _sinhx_x(x_2)
        P_c = P.mean(axis=0)
        F_c = F.mean(axis=0)
        # Centroid: classical exp_iL2.
        P_c_new = P_c * jnp.exp(-x) + dt_2 * F_c * sinhP * jnp.exp(-x_2)
        # Internal modes: simple B kick.
        P_int_new = (P - P_c[None]) + dt_2 * (F - F_c[None])
        return P_c_new[None] + P_int_new

    def exp_iL1_pimd(box, R, V, V_b, dt_step, **kwargs):
        """Position drift with PR scaling on centroid only.
           dR_k = dt_step * V_k + (centroid PR adjustment, broadcast across beads).
        """
        V_b_per_axis = membership @ V_b
        x = V_b_per_axis * dt_step
        x_2 = x / 2
        sinhV = _sinhx_x(x_2)
        R_c = R.mean(axis=0)
        V_c = V.mean(axis=0)
        # Adjustment lifts free drift to PR drift on the centroid:
        #   centroid_pr - centroid_free = R_c*(exp(x)-1) + dt*V_c*(exp(x/2)*sinh - 1)
        adjustment = (R_c * (jnp.exp(x) - 1)
                      + dt_step * V_c * (jnp.exp(x_2) * sinhV - 1.0))
        dR = dt_step * V + adjustment[None]
        return shift_fn(R, dR, box=box, **kwargs)

    def init_fn(key, R, box, mass=f32(1.0), **kwargs):
        N = R.shape[1]
        dim = R.shape[2]
        _kT_eff = kwargs.get('kT', kT_eff)

        if jnp.isscalar(box) or box.ndim == 0:
            box = jnp.eye(dim) * box

        box_position = jnp.zeros(G, dtype=R.dtype)
        box_momentum = jnp.zeros(G, dtype=R.dtype)
        # Box mass tied to the physical (not effective) temperature.
        box_mass = jnp.asarray(n_per_group * (N + 1) * kT_phys * tau_p ** 2,
                               dtype=R.dtype)

        _, F, dUdV = force_full_and_dUdV(R, box,
                                         **{k: v for k, v in kwargs.items() if k != 'kT'})
        key, split = jax.random.split(key)

        state = PIMDNPTLangevinState(
            position=R, momentum=None, force=F, mass=mass,
            reference_box=box,
            box_position=box_position, box_momentum=box_momentum, box_mass=box_mass,
            dUdV=dUdV, group_ids=group_ids, rng=key,
        )
        state = _jms.canonicalize_mass(state)
        return _jms.initialize_momenta(state, split, _kT_eff)

    @jax.jit
    def step_fn(state, **kwargs):
        _kT_eff = kwargs.pop('kT', kT_eff)
        _P_target = kwargs.pop('pressure', pressure)

        R, P, M, F = state.position, state.momentum, state.mass, state.force
        R_b, P_b, M_b = state.box_position, state.box_momentum, state.box_mass
        ref_box = state.reference_box
        dUdV = state.dUdV

        N = R.shape[1]
        dim = R.shape[2]
        N_f = f32(dim * N)

        # 1. Half barostat kick.
        _, vol = _box_and_vol(R_b, ref_box)
        G_e = box_force_cv(vol, dUdV, P.mean(axis=0), M, _P_target, N_f)
        P_b = P_b + dt_2 * G_e

        # 2. Half momentum kick (B with PR on centroid).
        P = exp_iL2_pimd(P, F, P_b / M_b, N_f)

        # 3. Half box-position + half position drift (A).
        R_b = R_b + dt_2 * P_b / M_b
        box, _ = _box_and_vol(R_b, ref_box)
        R = exp_iL1_pimd(box, R, P / M, P_b / M_b, dt_2, **kwargs)

        # 4. O step: PILE-L on bead momenta + Langevin on box momentum.
        rng = state.rng
        rng, sub1, sub2 = jax.random.split(rng, 3)
        # PILE-L:
        P_nm = jnp.tensordot(nm_trans.T, P, axes=(1, 0))
        c1 = jnp.exp(-gamma_P * dt)[:, None, None]
        c2 = jnp.sqrt(jnp.maximum(_kT_eff * (1.0 - c1**2), 0.0))
        noise = jax.random.normal(sub1, shape=P_nm.shape, dtype=P_nm.dtype)
        P_nm = c1 * P_nm + c2 * jnp.sqrt(M) * noise
        P = jnp.tensordot(nm_trans, P_nm, axes=(1, 0))
        # Langevin barostat at kT_phys:
        c1_b = jnp.exp(-gamma_box * dt)
        c2_b = jnp.sqrt(jnp.maximum(kT_phys * (1.0 - c1_b**2), 0.0))
        noise_b = jax.random.normal(sub2, shape=P_b.shape, dtype=P_b.dtype)
        P_b = c1_b * P_b + c2_b * jnp.sqrt(M_b) * noise_b

        # 5. Half box-position + half position drift (A).
        R_b = R_b + dt_2 * P_b / M_b
        box, vol = _box_and_vol(R_b, ref_box)
        R = exp_iL1_pimd(box, R, P / M, P_b / M_b, dt_2, **kwargs)

        # 6. Force evaluation at new (R, box).
        _, F, dUdV = force_full_and_dUdV(R, box, **kwargs)

        # 7. Half momentum kick (B with PR on centroid).
        P = exp_iL2_pimd(P, F, P_b / M_b, N_f)

        # 8. Half barostat kick.
        G_e = box_force_cv(vol, dUdV, P.mean(axis=0), M, _P_target, N_f)
        P_b = P_b + dt_2 * G_e

        return state.set(
            position=R, momentum=P, force=F,
            box_position=R_b, box_momentum=P_b,
            dUdV=dUdV, rng=rng,
        )

    return init_fn, step_fn
