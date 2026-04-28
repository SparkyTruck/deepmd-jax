"""Simulation-only helpers that depend on jax_md.

Split out of md.py so that the training path (which imports utils.py) does not
pull in jax_md. Everything here is internal infrastructure for Simulation.
"""
import os
from typing import Callable

import jax
import jax.numpy as jnp
import jax_md
import numpy as np
from ase import Atoms, io
from jax.sharding import PartitionSpec as PSpec

from .utils import split, reorder_by_device, get_mask_by_device


def iso_pressure(energy_fn, position, box, n_atoms, kT, **kwargs):
    """Isotropic pressure (eV/Å^3) from a uniform-strain derivative.

    P = (N kT - dU/dε / 3) / V, where dU/dε is taken under
    perturbation=(1+ε) on both real-space coord and box. PBC-correct since
    the energy_fn handles the box internally.

    For classical MD pass the full ``energy_fn``. For PIMD pass the bead-
    averaged ML-only energy (the spring is V-independent in real coords and
    must not enter the pressure).

    The instantaneous-KE form ``(2KE - dU/dε) / (3V)`` would be equivalent at
    equilibrium for classical MD; using N kT directly gives the same
    expectation with smaller per-step noise, and is the centroid-virial
    estimator for PIMD.
    """
    def U(eps):
        return energy_fn(position, box=box, perturbation=(1 + eps), **kwargs)
    dUdε = jax.grad(U)(jnp.zeros((), position.dtype))
    V = jnp.linalg.det(box) if box.ndim == 2 else jnp.prod(box)
    return (n_atoms * kT - dUdε / 3.0) / V


def min_image_unwrap(previous_position, current_position, box):
    '''Unwrap a step's real-space positions against the previous step,
    undoing the box-length jumps the periodic shift_fn may have introduced.
    '''
    if box.size == 3:
        return previous_position + (current_position - previous_position + box/2) % box - box/2
    delta_frac = (current_position - previous_position) @ jnp.linalg.inv(box)
    return previous_position + (delta_frac - jnp.round(delta_frac)) @ box


def print_run_profile(steps, elapsed_time, natoms, dt):
    '''Per-run timing summary.'''
    steps_per_microsecond_per_atom = 1e-6 * steps * natoms / (elapsed_time + 1e-6)
    nanosecond_per_day = 1e-6 * steps * dt * (86400 / elapsed_time)
    print('# Finished %d steps in %dh %dm %ds.' %
          (steps, elapsed_time // 3600, (elapsed_time % 3600) // 60, elapsed_time % 60))
    print('# Performance: %.3f ns/day, %.3f step/μs/atom' %
          (nanosecond_per_day, steps_per_microsecond_per_atom))


def prepare_dumps(requested, type_idx, type_symbols, n_bead, default_interval,
                  dump_interval=None):
    '''Validate dump kwargs and build a list of (kind, path, ase.Atoms) entries.

    requested: ``{'position'|'velocity'|'centroid': path or None}``.
    Returns ``([], None)`` if nothing was requested. Each existing dump file is
    truncated so every run() call produces its own output.
    '''
    requested = {k: v for k, v in requested.items() if v is not None}
    if not requested:
        return [], None
    if type_symbols is None:
        raise ValueError(
            "Simulation.run(dump_*=...) requires `type_symbols=[...]` at "
            "Simulation(...) construction so XYZ output can label atoms.")
    is_pimd = n_bead > 1
    if 'centroid' in requested and not is_pimd:
        raise ValueError("dump_centroid is only meaningful when n_bead > 1.")
    interval = int(dump_interval) if dump_interval is not None else int(default_interval)
    if interval <= 0:
        raise ValueError(f"dump_interval must be positive; got {interval}")
    symbols_N = np.array(type_symbols, dtype=object)[type_idx]
    symbols_PN = np.tile(symbols_N, n_bead) if is_pimd else symbols_N
    dumps = []
    for kind, path in requested.items():
        # Per-bead position/velocity dumps flatten (P, N, 3) to (P*N, 3).
        flatten = is_pimd and kind != 'centroid'
        syms = symbols_PN if flatten else symbols_N
        atoms = Atoms(symbols=list(syms), positions=np.zeros((len(syms), 3)), pbc=True)
        if os.path.exists(path):
            os.remove(path)
        dumps.append((kind, path, atoms))
    return dumps, interval


def write_dump_frame(dumps, pos_frame, vel_frame, centroid_frame, box_frame):
    '''Append one frame to each dump file.'''
    if box_frame.shape == (3,):
        cell = np.concatenate([np.array(box_frame), [90, 90, 90]])
    else:
        cell = np.array(box_frame)
    sources = {'position': pos_frame, 'velocity': vel_frame, 'centroid': centroid_frame}
    for kind, path, atoms in dumps:
        data = np.asarray(sources[kind])
        if data.ndim == 3:
            data = data.reshape(-1, 3)
        atoms.set_positions(data)
        atoms.set_cell(cell)
        io.write(path, atoms, append=True)


def stack_typed_nbrs_per_bead(typed_nbr_fn, real_pos_PN3, box, n_bead):
    '''Build one stacked TypedNeighborList covering all P beads.

    Allocates each bead with a shared knbr (the per-bead max), then uses
    `update` for replicas 1..P-1 so all replicas share `allocate`'s closures
    and the resulting pytrees can be stacked along a leading P axis.
    '''
    # Suppress per-bead "Neighborlist allocated" log lines; print one summary
    # below once the shared capacity is fixed.
    import builtins
    _orig_print = builtins.print
    builtins.print = lambda *a, **k: None
    try:
        per_bead_tmp = [typed_nbr_fn.allocate(real_pos_PN3[k], box=box)
                        for k in range(n_bead)]
    finally:
        builtins.print = _orig_print
    knbr_shared = [int(max(nb.knbr[i] for nb in per_bead_tmp))
                   for i in range(len(per_bead_tmp[0].knbr))]
    nbrs_0 = typed_nbr_fn.allocate(real_pos_PN3[0], box=box,
                                   knbr_override=knbr_shared)
    per_bead = [nbrs_0] + [
        typed_nbr_fn.update(real_pos_PN3[k], nbrs_0, box=box)
        for k in range(1, n_bead)
    ]
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *per_bead)


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
            idx = jax.lax.with_sharding_constraint(idx, PSpec('atom'))
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
        idx = jax.lax.with_sharding_constraint(idx, PSpec('atom'))
        filter = full_mask[:,None] * jnp.isin(idx, idx_mask_out, invert=True)
        return jnp.where(filter, idx, len(full_mask))
    return idx_mask_fn


@jax_md.dataclasses.dataclass
class TypedNeighborList():
    '''
        Wraps jax_md.partition.NeighborList with typed lists.
    '''
    nbrs: jax_md.partition.NeighborList
    knbr: list = jax_md.dataclasses.static_field()
    nbrs_nm: list
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


def typed_neighbor_list(box, type_idx, type_count, rcut, buffer_ratio=1.2):
    '''
        Returns a typed neighbor list function that can be used to allocate and update neighbor lists.
        allocate_fn() and update_fn() accept real space coordinates but processes internally in fractional coordinates.
    '''
    type_idx = np.array(type_idx, dtype=int)
    type_count = tuple(type_count)
    reference_box = box.astype(jnp.float32) # box at creation of typed_neighbor_list_fn; pass only this box to jax_md.neighbor_list
    type_mask_fns = get_type_mask_fns(type_count)
    idx_mask_fn = get_idx_mask_fn(type_count)

    def canonicalize(coord, box=reference_box):
        # scale coord to reference_box and store in float32 fractional coordinates
        box = box.astype(jnp.float32)
        scale = reference_box / box  # (3,) per-axis; works for iso and semi_iso
        coord = (coord.astype(jnp.float32) % box) * scale
        # shrink by 1e-7 from the boundary to avoid numerical issues
        coord = coord * (1-5e-7)
        # sort by type, pad each type, and partition to multiple devices
        coord = reorder_by_device(coord, type_idx)
        return coord

    # allocate function: non-jit-compatible
    def allocate_fn(coord, box=reference_box, knbr_override=None):
        '''
            If knbr_override (a list matching the len(type_count) of per-type neighbor
            capacities) is provided, skip the measurement step and build the neighbor
            list at that capacity. Used by PIMD to allocate multiple replicas that
            share a common layout.
        '''
        coord = canonicalize(coord, box)
        displacement_fn = jax_md.space.periodic(reference_box)[0]
        # Measure the exact size of max neighbors for each atom type (still needed
        # to size the underlying jax_md neighbor list).
        test_nbr = jax_md.partition.neighbor_list(displacement_fn,
                                                  reference_box,
                                                  rcut * (reference_box / box).max(), # worst-case per-axis shrink
                                                  capacity_multiplier=1.,
                                                  custom_mask_function=idx_mask_fn
                                                ).allocate(coord)
        if knbr_override is None:
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
            print(f'# Neighborlist allocated with size {np.array(knbr) - 1}, rcut_plus_skin = {rcut}, buffer_ratio = {buffer_ratio}')
        else:
            knbr = list(int(k) for k in knbr_override)
        # infer a total buffer from the max neighbors of each type
        total_buffer_ratio = (sum(knbr)+1.01) / test_nbr.idx.shape[1]
        # Allocate the neighbor list with the inferred buffer ratio
        nbrs = jax_md.partition.neighbor_list(displacement_fn,
                                              box,
                                              rcut * (reference_box / box).max(), # worst-case per-axis shrink
                                              capacity_multiplier=total_buffer_ratio * 1.1,
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
        nbr_idx = jax.lax.with_sharding_constraint(nbrs.idx, PSpec('atom'))
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
        nbrs_nm = [[n[:,:-1] for n in nbrs_m] for nbrs_m in nbrs_nm]
        return nbrs_nm, overflow

    return typed_neighbor_list_fn(allocate=allocate_fn, update=update_fn)
