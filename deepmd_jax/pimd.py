"""Path-integral (ring-polymer) molecular dynamics support.

Provides a BAOAB Langevin integrator that mirrors ``jax_md.simulate.nvt_langevin``
but operates on ring-polymer states shaped ``(P, N, 3)`` and replaces the O step
with the PILE-L thermostat (Langevin on normal-mode momenta).

The ring-polymer spring is added inside the integrator on top of the
user-supplied (bead-averaged) potential, so the B step looks exactly like the
classical Langevin case. The price is a dt limit of roughly 1/omega_max where
omega_max ~ 2 * n_bead * k_B T / hbar.
"""

import jax
import jax.numpy as jnp
import numpy as np
import jax_md
from jax_md import simulate

HBAR = 6.582119569e-16 * 1e15  # reduced Planck's constant, eV*fs


def normal_mode_transform(n_bead, kT):
    """Normal-mode frequencies and orthogonal transform for a ring polymer.

    Returns
    -------
    freqs : np.ndarray, shape ``(P,)``
        Normal-mode angular frequencies in fs^-1. ``freqs[0]`` is 0 (centroid).
    C : np.ndarray, shape ``(P, P)``
        Real orthogonal matrix mapping normal-mode to primitive coordinates:
        ``R_primitive = C @ R_nm``; inversely ``R_nm = C.T @ R_primitive``.
    """
    ring_poly_freq = n_bead * kT / HBAR
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
    thermostat_work : scalar; cumulative work done by the PILE-L thermostat
        (sum of ``KE_after_O - KE_before_O`` across O steps). Subtracting this
        from the ring-polymer Hamiltonian yields a quantity conserved up to
        BA-step discretization error — the Langevin analogue of the NVE
        invariant.
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


def initial_pimd_momentum(R, mass, kT, key):
    """Maxwell-Boltzmann sample at the ring-polymer temperature P·kT.
       R: (P, N, 3); mass: (N, 1)."""
    n_bead = R.shape[0]
    return jnp.sqrt(mass * kT * n_bead) * jax.random.normal(
        key, R.shape, dtype=R.dtype)


def spring_energy(R, mass, box, omega_P):
    """Ring-polymer spring energy in real coords:
        E = 0.5 * m * omega_P^2 * sum_α (R_α - R_{α-1})^2.
    R    : (P, N, 3); mass: (N, 1); box: (3,) or (3, 3); omega_P: scalar (fs^-1).
    """
    R_uw = unwrap_across_beads(R, box)
    dR = R_uw - jnp.roll(R_uw, 1, axis=0)
    return 0.5 * omega_P ** 2 * jnp.sum(mass * (dR ** 2))


def _pile_l_stochastic_step(state, dt, kT, gamma_P, nm_trans):
    """O step in the normal-mode basis.

    gamma_P : ``(P,)`` per-mode friction.
    nm_trans : ``(P, P)`` orthogonal transform ``C``.
    """
    n_bead = state.position.shape[0]
    P_old = state.momentum
    P_nm = jnp.tensordot(nm_trans.T, P_old, axes=(1, 0))             # (P, N, 3)
    c1 = jnp.exp(-gamma_P * dt)[:, None, None]                       # (P, 1, 1)
    c2 = jnp.sqrt(jnp.maximum(kT * n_bead * (1.0 - c1**2), 0.0))     # (P, 1, 1)
    key, split = jax.random.split(state.rng)
    noise = jax.random.normal(split, shape=P_nm.shape, dtype=P_nm.dtype)
    P_nm = c1 * P_nm + c2 * jnp.sqrt(state.mass) * noise
    momentum = jnp.tensordot(nm_trans, P_nm, axes=(1, 0))
    delta_KE = 0.5 * jnp.sum((momentum ** 2 - P_old ** 2) / state.mass)
    return state.set(
        momentum=momentum, rng=key,
        thermostat_work=state.thermostat_work + delta_KE,
    )


def nvt_langevin_pimd(
    energy_fn,
    shift_fn,
    dt,
    kT,
    box,
    gamma_P,
    nm_trans,
    **sim_kwargs,
):
    """BAOAB Langevin integrator for PIMD with a PILE-L thermostat.

    Mirrors ``jax_md.simulate.nvt_langevin``; only the O (stochastic) step
    differs. The ring-polymer spring is added internally; the caller passes the
    bead-averaged potential only.

    Args
    ----
    energy_fn : callable
        ``energy_fn(R, **kwargs) -> scalar`` with ``R`` of shape ``(P, N, 3)``;
        the bead-averaged potential to which the ring-polymer spring is added.
    shift_fn : callable
        Shift applied by ``position_step``; must broadcast over a leading bead
        axis (jax_md's ``periodic`` / ``periodic_general`` do).
    dt : float, time step in fs.
    kT : float, temperature (eV).
    box : jnp.ndarray, the (fixed) simulation cell, used by the spring term.
    gamma_P : jnp.ndarray, shape ``(P,)``
        Per-normal-mode friction coefficient in fs^-1.
    nm_trans : jnp.ndarray, shape ``(P, P)``
        Orthogonal transform ``C`` (primitive <- normal-mode).
    """
    n_bead = nm_trans.shape[0]
    omega_P = n_bead * kT / HBAR

    def total_energy_fn(R, mass, **kwargs):
        kwargs.pop('box', None)
        return (n_bead * energy_fn(R, box=box, **kwargs)
                + spring_energy(R, mass, box, omega_P))

    def total_force_fn(R, mass, **kwargs):
        return -jax.grad(total_energy_fn)(R, mass, **kwargs)

    @jax.jit
    def init_fn(key, R, mass=jnp.float32(1.0), **kwargs):
        _kT = kwargs.pop('kT', kT)
        key, split = jax.random.split(key)
        state = PIMDLangevinState(
            position=R, momentum=None, force=None, mass=mass, rng=key,
            thermostat_work=jnp.zeros((), dtype=R.dtype),
        )
        state = simulate.canonicalize_mass(state)
        state = state.set(force=total_force_fn(R, state.mass, **kwargs))
        return state.set(momentum=initial_pimd_momentum(R, state.mass, _kT, split))

    @jax.jit
    def step_fn(state, **kwargs):
        _dt = kwargs.pop('dt', dt)
        _kT = kwargs.pop('kT', kT)
        dt_2 = _dt / 2

        state = simulate.momentum_step(state, dt_2)
        state = simulate.position_step(state, shift_fn, dt_2, **kwargs)
        state = _pile_l_stochastic_step(state, _dt, _kT, gamma_P, nm_trans)
        state = simulate.position_step(state, shift_fn, dt_2, **kwargs)
        state = state.set(force=total_force_fn(state.position, state.mass, **kwargs))
        state = simulate.momentum_step(state, dt_2)
        return state

    return init_fn, step_fn


def unwrap_across_beads(coord, box):
    """Minimum-image-shift every bead onto the same periodic image as bead 0.
        coord : jnp.ndarray, shape ``(P, N, 3)``
        box : jnp.ndarray, shape ``(3,)`` or ``(3, 3)``
    Returns
        coord_consistent : jnp.ndarray, shape ``(P, N, 3)``
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


def unwrap_step_position(previous_position, current_position, box):
    """Put ``current_position`` on the nearest periodic image to ``previous_position``."""
    if box.size == 3:
        return (previous_position
                + (current_position - previous_position + box / 2) % box
                - box / 2)
    delta_frac = (current_position - previous_position) @ jnp.linalg.inv(box)
    return previous_position + (delta_frac - jnp.round(delta_frac)) @ box


def centroid(coord, box):
    """Ring-polymer centroid with correct handling of periodic wrapping.

    Equivalent to ``unwrap_across_beads(coord, box).mean(axis=0)`` — included as
    a top-level helper so users can lift a trajectory back to centroid space
    without reimplementing the unwrap.
    """
    return unwrap_across_beads(coord, box).mean(axis=0)


# -----------------------------------------------------------------------------
# NPT-PIMD: Parrinello-Rahman on the centroid mode + Langevin barostat.
# -----------------------------------------------------------------------------


@jax_md.dataclasses.dataclass
class PIMDNPTLangevinState:
    """State for BAOAB Langevin PIMD with PILE-L thermostat and a Langevin barostat.

    Position/momentum/force are ``(P, N, 3)``; box DOF is ``(G,)`` per couple-axes
    group. ``dUdV`` here is the bead-averaged ``dU/d_eps`` centroid-virial
    strain derivative of the user-supplied potential only; the spring term is
    independent of the cell in real coords and never enters the barostat.
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

    thermostat_work: jnp.ndarray
    box_thermostat_work: jnp.ndarray

    @property
    def velocity(self):
        return self.momentum / self.mass

    @property
    def box(self):
        scales_g = jnp.exp(self.box_position)
        safe = jnp.maximum(self.group_ids, 0)
        per_axis = jnp.where(self.group_ids >= 0, scales_g[safe], 1.)
        return self.reference_box * per_axis


def _sinhx_x(x):
    return (1 + x ** 2 / 6 + x ** 4 / 120 + x ** 6 / 5040
            + x ** 8 / 362_880 + x ** 10 / 39_916_800)


def npt_langevin_pimd(
    energy_fn,
    shift_fn,
    dt,
    *,
    pressure,
    kT,
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
    Langevin barostat couples only to the centroid. The ring-polymer spring is
    added internally; the caller passes the bead-averaged potential only.

    In real coords the spring depends only on bead positions (not the cell),
    so it never enters dU/dV. The potential virial is the centroid-virial
    derivative: the centroid and box are strained together, while bead
    displacements from the centroid are held fixed.
    """
    f32 = jnp.float32
    dt = f32(dt)
    dt_2 = f32(dt / 2)
    n_bead = nm_trans.shape[0]
    omega_P = n_bead * kT / HBAR
    G = n_per_group.shape[0]
    membership = jnp.asarray(membership)
    n_per_group = jnp.asarray(n_per_group, dtype=f32)
    group_ids = jnp.asarray(group_ids)

    def force_stress_fn(R, box, **kwargs):
        """Returns (F_per_bead, dU/d_eps_per_group) from the user potential.

        The centroid-virial strain derivative is evaluated as the ordinary
        full-coordinate strain derivative plus the internal-mode virial
        correction, so only centroid separations contribute to the cell force.
        """
        def U(R_, eps):
            perturbation = 1.0 + membership @ eps
            return n_bead * energy_fn(R_, box=box, perturbation=perturbation,
                                      **kwargs)
        eps0 = jnp.zeros(G, dtype=R.dtype)
        _, grads = jax.value_and_grad(U, argnums=(0, 1))(R, eps0)
        F = -grads[0]
        R_uw = unwrap_across_beads(R, box)
        dR_internal = R_uw - R_uw.mean(axis=0, keepdims=True)
        internal_virial_axis = jnp.sum(dR_internal * F, axis=(0, 1))
        return F, (grads[1] + membership.T @ internal_virial_axis) / n_bead

    def force_and_dUdV(R, mass, box, **kwargs):
        """Bead force from (energy_fn + spring); dUdV is from energy_fn only."""
        F, dUdV = force_stress_fn(R, box, **kwargs)
        F_spring = -jax.grad(spring_energy)(R, mass, box, omega_P)
        return F + F_spring, dUdV

    def _box_and_vol(R_b, ref_box):
        scales_g = jnp.exp(R_b)
        per_axis = membership @ scales_g + (1.0 - membership.sum(axis=1))
        box = ref_box * per_axis
        V = jnp.linalg.det(box) if box.ndim == 2 else jnp.prod(box)
        return box, V

    def _per_axis_rate(vel_b, N_f):
        vel_b_per_axis = membership @ vel_b
        tr_vel_b = jnp.sum(n_per_group * vel_b)
        return vel_b_per_axis + tr_vel_b / N_f

    def box_force_cv(V, dUdV, P_centroid, mass, P_target, N_f):
        """Box-momentum driver."""
        KE2_axis = jnp.sum(P_centroid ** 2 / mass, axis=0)            # (3,)
        KE2_group = membership.T @ KE2_axis                            # (G,)
        KE2_total = jnp.sum(KE2_axis)
        return (KE2_group + (n_per_group / N_f) * KE2_total
                - dUdV - P_target * V * n_per_group)

    def exp_iL2_pimd(P, F, vel_b, N_f):
        """Half-step momentum update with PR rescale on centroid mode only."""
        rate = _per_axis_rate(vel_b, N_f)
        x = rate * dt_2
        x_2 = x / 2
        sinh_factor = _sinhx_x(x_2)
        P_c = P.mean(axis=0)
        F_c = F.mean(axis=0)
        # Centroid: classical exp_iL2.
        P_c_new = P_c * jnp.exp(-x) + dt_2 * F_c * sinh_factor * jnp.exp(-x_2)
        # Internal modes: simple B kick.
        P_int_new = (P - P_c[None]) + dt_2 * (F - F_c[None])
        return P_c_new[None] + P_int_new

    def exp_iL1_pimd(box, R, vel, vel_b, dt_step, **kwargs):
        """Position drift with PR scaling on centroid only.
           dR_k = dt_step * vel_k + (centroid PR adjustment, broadcast across beads).
        """
        vel_b_per_axis = membership @ vel_b
        x = vel_b_per_axis * dt_step
        x_2 = x / 2
        sinh_factor = _sinhx_x(x_2)
        # Use the same periodic-image convention as the centroid-virial stress.
        # The periodic shift_fn can wrap beads between the two half drifts, and
        # a raw bead mean would move the wrong centroid when a ring polymer
        # straddles a boundary.
        R_c = unwrap_across_beads(R, box).mean(axis=0)
        vel_c = vel.mean(axis=0)
        # Adjustment lifts free drift to PR drift on the centroid:
        #   centroid_pr - centroid_free = R_c*(exp(x)-1) + dt*vel_c*(exp(x/2)*sinh - 1)
        adjustment = (R_c * (jnp.exp(x) - 1)
                      + dt_step * vel_c * (jnp.exp(x_2) * sinh_factor - 1.0))
        dR = dt_step * vel + adjustment[None]
        return unwrap_step_position(R, shift_fn(R, dR, box=box, **kwargs), box)

    def init_fn(key, R, box, mass=f32(1.0), **kwargs):
        N = R.shape[1]
        dim = R.shape[2]
        _kT = kwargs.get('kT', kT)

        if jnp.isscalar(box) or box.ndim == 0:
            box = jnp.eye(dim) * box

        box_position = jnp.zeros(G, dtype=R.dtype)
        box_momentum = jnp.zeros(G, dtype=R.dtype)
        box_mass = jnp.asarray(n_per_group * (N + 1) * _kT * tau_p ** 2,
                               dtype=R.dtype)

        key, split = jax.random.split(key)
        state = PIMDNPTLangevinState(
            position=R, momentum=None, force=None, mass=mass,
            reference_box=box,
            box_position=box_position, box_momentum=box_momentum, box_mass=box_mass,
            dUdV=None, group_ids=group_ids, rng=key,
            thermostat_work=jnp.zeros((), dtype=R.dtype),
            box_thermostat_work=jnp.zeros((), dtype=R.dtype),
        )
        state = simulate.canonicalize_mass(state)
        F, dUdV = force_and_dUdV(R, state.mass, box,
                                 **{k: v for k, v in kwargs.items() if k != 'kT'})
        state = state.set(force=F, dUdV=dUdV)
        return state.set(momentum=initial_pimd_momentum(R, state.mass, _kT, split))

    @jax.jit
    def step_fn(state, **kwargs):
        _kT = kwargs.pop('kT', kT)
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
        P_old = P
        P_nm = jnp.tensordot(nm_trans.T, P, axes=(1, 0))
        c1 = jnp.exp(-gamma_P * dt)[:, None, None]
        c2 = jnp.sqrt(jnp.maximum(_kT * n_bead * (1.0 - c1**2), 0.0))
        noise = jax.random.normal(sub1, shape=P_nm.shape, dtype=P_nm.dtype)
        P_nm = c1 * P_nm + c2 * jnp.sqrt(M) * noise
        P = jnp.tensordot(nm_trans, P_nm, axes=(1, 0))
        delta_KE = 0.5 * jnp.sum((P ** 2 - P_old ** 2) / M)
        # Langevin barostat:
        P_b_old = P_b
        c1_b = jnp.exp(-gamma_box * dt)
        c2_b = jnp.sqrt(jnp.maximum(_kT * (1.0 - c1_b**2), 0.0))
        noise_b = jax.random.normal(sub2, shape=P_b.shape, dtype=P_b.dtype)
        P_b = c1_b * P_b + c2_b * jnp.sqrt(M_b) * noise_b
        delta_KE_box = 0.5 * jnp.sum((P_b ** 2 - P_b_old ** 2) / M_b)

        # 5. Half box-position + half position drift (A).
        R_b = R_b + dt_2 * P_b / M_b
        box, vol = _box_and_vol(R_b, ref_box)
        R = exp_iL1_pimd(box, R, P / M, P_b / M_b, dt_2, **kwargs)

        # 6. Force evaluation at new (R, box).
        F, dUdV = force_and_dUdV(R, M, box, **kwargs)

        # 7. Half momentum kick (B with PR on centroid).
        P = exp_iL2_pimd(P, F, P_b / M_b, N_f)

        # 8. Half barostat kick.
        G_e = box_force_cv(vol, dUdV, P.mean(axis=0), M, _P_target, N_f)
        P_b = P_b + dt_2 * G_e

        return state.set(
            position=R, momentum=P, force=F,
            box_position=R_b, box_momentum=P_b,
            dUdV=dUdV, rng=rng,
            thermostat_work=state.thermostat_work + delta_KE,
            box_thermostat_work=state.box_thermostat_work + delta_KE_box,
        )

    return init_fn, step_fn
