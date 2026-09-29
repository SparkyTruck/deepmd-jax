import jax
import optax
import numpy as np
import jax.numpy as jnp
import time, datetime
import flax.linen as nn
from functools import partial
from .utils import (get_p3mlr_fn, get_p3mlr_grid_size, load_model, save_model,
                    compress_model, dplr_charges, get_max_nbrs,
                    neighborlist_is_efficient, _atomic_write_bytes,
                    _write_sha256_sidecar, _verify_sha256_sidecar)
from .data import Dataset, compute_lattice_candidate, get_atomic_scalar_stats
from .dpmodel import DPModel
from typing import Union, List
import tempfile
import os
import hashlib
import json
import pickle


_CHECKPOINT_VERSION = 1

# Neighbor-derived statistics are reduced through JAX kernels and can differ
# slightly between CPU and GPU backends even when the dataset and sampling
# trajectory are identical.  The portable values remain the actual model
# contract; these tolerances only validate an independently recomputed guard.
_PORTABLE_DERIVED_TOLERANCES = {
    'Ebias': (1e-6, 1e-8),
    'sr_mean': (2e-4, 1e-8),
    'sr_std': (2e-4, 1e-8),
    'Nnbrs': (2e-4, 1e-8),
}


def _path_fingerprint(paths):
    if paths is None:
        return None
    if isinstance(paths, (str, os.PathLike)):
        return os.path.abspath(os.fspath(paths))
    return [_path_fingerprint(path) for path in paths]


def _tree_to_host(tree):
    return jax.tree_util.tree_map(
        lambda value: np.asarray(value) if hasattr(value, 'shape') else value,
        tree)


def _save_training_checkpoint(path, payload):
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_write_bytes(path, raw)
    _write_sha256_sidecar(path, hashlib.sha256(raw).hexdigest())


def _load_training_checkpoint(path):
    _verify_sha256_sidecar(path, required=True)
    with open(path, 'rb') as file:
        payload = pickle.load(file)
    if payload.get('checkpoint_version') != _CHECKPOINT_VERSION:
        raise ValueError('Unsupported training checkpoint version.')
    return payload


def _write_history(path, history):
    raw = (json.dumps(history, indent=2, sort_keys=True) + '\n').encode('utf-8')
    _atomic_write_bytes(path, raw)
    _write_sha256_sidecar(path, hashlib.sha256(raw).hexdigest())


def _contract_values_equal(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return (set(left) == set(right)
                and all(_contract_values_equal(left[key], right[key])
                        for key in left))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return (len(left) == len(right)
                and all(_contract_values_equal(a, b)
                        for a, b in zip(left, right)))
    if hasattr(left, 'shape') or hasattr(right, 'shape'):
        try:
            return np.array_equal(np.asarray(left), np.asarray(right))
        except Exception:
            return False
    return left == right


def _load_portable_model_params(path, computed_params):
    """Load exact static params while verifying the local data-derived shape."""
    supplied_sha256 = _verify_sha256_sidecar(path, required=True)
    with open(path, 'rb') as file:
        supplied = pickle.load(file)
    if not isinstance(supplied, dict) or set(supplied) != set(computed_params):
        raise ValueError('Portable model params have incompatible fields.')
    for key in supplied:
        if key in _PORTABLE_DERIVED_TOLERANCES:
            left, right = np.asarray(supplied[key]), np.asarray(computed_params[key])
            rtol, atol = _PORTABLE_DERIVED_TOLERANCES[key]
            if (left.shape != right.shape or left.dtype != right.dtype
                    or not np.allclose(left, right, rtol=rtol, atol=atol)):
                raise ValueError(
                    'Portable model params disagree with local data statistics: '
                    '%s (rtol=%g, atol=%g).' % (key, rtol, atol))
        elif not _contract_values_equal(supplied[key], computed_params[key]):
            raise ValueError(
                'Portable model params disagree with the model contract: %s.' % key)
    print('# Loaded content-addressed portable model params from \'%s\'.' % path)
    return supplied, supplied_sha256

def _get_static_args(type_idx, lattice_args):
    return nn.FrozenDict({'type_idx': tuple(type_idx),
                          'lattice': lattice_args,
                          'use_neighborlist': lattice_args['use_neighborlist'],
                          'max_nbrs': lattice_args['max_nbrs']})

def train(
    model_type: str,
    rcut: float,
    train_data_path: Union[str, List[str]],
    val_data_path: Union[str, List[str]] = None,
    save_path: str = 'model.pkl',
    step: int = 1000000,
    mp: bool = False,
    atomic_sel: List[int] = None,
    embed_widths: List[int] = [32,32,64],
    embed_mp_widths: List[int] = [64,64,64],
    fit_widths: List[int] = None,
    axis_neurons: int=12,
    lr: float = None,
    batch_size: int = None,
    val_batch_size_ratio: int = 4,
    compress: bool = True,
    print_every: int = 1000,
    atomic_data_prefix: str = None,
    s_pref_e: float = 0.02,
    l_pref_e: float = 1,
    s_pref_f: float = 1000,
    l_pref_f: float = 1,
    dplr_wannier_model_path: str = None,
    dplr_q_atoms: List[float] = None,
    dplr_q_wc: List[float] = None,
    dplr_beta: float = 0.4,
    dplr_resolution: float = 5,
    lr_limit: float = 1e-6,
    beta2: float = 0.99,
    decay_steps: int = 5000,
    getstat_bs: int = 64,
    label_bs: int = 256,
    print_loss_smoothing: int = 20,
    compress_Ngrids: int = 1024,
    compress_r_min: float = 0.6,
    seed: int = None,
    loss: str = 'l1-mixed',
    hybrid: bool = False,
    obs_s_pref: float = 0.005,
    obs_l_pref: float = 0.1,
    obs_train_data_path: Union[str, List[str], List[List[str]]] = None,
    obs_batch_size: int = 8,
    obs_temperature: Union[float, list] = None,
    obs_target: Union[float, str, list, None] = None,
    obs_step_every: int = 1,
    use_neighbor_list_when_possible: bool = True,
    checkpoint_path: str = None,
    checkpoint_every: int = None,
    resume: bool = False,
    max_updates_per_run: int = None,
    history_path: str = None,
    model_params_path: str = None,
):
    '''
        Entry point for training deepmd-jax models.

        Input arguments:
            model_type:
                 'energy' (standard force field),
                 'atomic' (predicts per-atom 3-vectors, e.g. Wannier centroid),
                 'atomic_t2' (predicts per-atom symmetric 3x3 tensors, e.g. polarizability),
                 'atomic_scalar' (predicts one invariant scalar per selected atom),
                 'dplr' (force field w/ long-range electrostatics).
            rcut: cutoff radius (Angstrom) for the model.
            save_path: path to save the trained model.
            train_data: path to training data (str) or list of paths to training data (List[str]).
            val_data: path to validation data (str) or list of paths to validation data (List[str]).
            step: number of training steps. Depending on dataset size, expect 1e5-1e7 for energy models and 1e5-1e6 for wannier models.
            mp: whether to use message passing model for more accuracy at a higher cost.
            atomic_sel: Selects the atom types for prediction. Must be provided for atomic models.
            embed_widths: Widths of the embedding neural network.
            embed_mp_widths: Widths of the embedding neural network in message passing. Only used when mp == True.
            fit_widths: Widths of the fitting neural network.
            axis_neurons: Number of axis neurons to project the atomic features before the fitting network. Recommended range: 8-16.
            lr: learning rate at start. If None, defaults to 0.002 for energy and 0.01 for atomic models.
            batch_size: training batch size in number of frames. If None, will be automatically determined by label_bs.
            val_batch_size_ratio: validation batch size / batch_size. Increase for stabler validation loss.
            compress: whether to compress the model after training for faster inference.
            print_every: interval for printing loss and validation.
            atomic_data_prefix: prefix for .npy label files. Defaults to 'atomic_dipole', 'atomic_polarizability', or 'atomic_energy'.
            s_pref_e: starting prefactor for energy loss.
            l_pref_e: limit prefactor for energy loss.
            s_pref_f: starting prefactor for force loss.
            l_pref_f: limit prefactor for force loss.
            dplr_wannier_model_path: path to the Deep Wannier model, only used in 'dplr'.
            dplr_q_atoms: charge of atomic cores for each atom type, only used in 'dplr'.
            dplr_q_wc: charge of wannier center/centroid for each type in atomic_sel of the wannier model, only used in 'dplr'.
            dplr_beta: inverse spread of the smoothed point charge distribution, only used in 'dplr'.
            dplr_resolution: higher resolution means denser grid: resolution = 1 / (grid length * beta); only used in 'dplr'.
            lr_limit: learning rate at end of training.
            beta2: adam optimizer parameter.
            decay_steps: learning rate exponentially decays every decay_steps.
            getstat_bs: batch size for computing model statistics at initialization.
            label_bs: training batch size in number of atoms.
            print_loss_smoothing: smoothing factor for loss printing.
            compress_Ngrids: Number of intervals used in compression.
            compress_r_min: A safe lower bound for interatomic distance in the compressed model.
            loss: loss function type, 'l1-mixed' or 'l2'.
                    'l1-mixed': MAE over configs/atoms, but RMS within each force/atomic vector; more robust to data outliers.
                    'l2': MSE over all entries. This was the old default.
            use_neighbor_list_when_possible: use a simple training neighborlist when the lattice candidate count is one.
            checkpoint_path: atomic training-state checkpoint path. Defaults to save_path + '.train.pkl' when checkpointing is enabled.
            checkpoint_every: save complete model/optimizer/RNG/sampler state every this many updates.
            resume: restore checkpoint_path and continue the exact optimizer and sampling trajectory.
            max_updates_per_run: optional segment cap; useful for sub-hour schedulers. The scientific target remains step.
            history_path: atomic JSON training history path. Defaults to save_path + '.history.json'.
            model_params_path: optional content-addressed pickle of exact static
                model parameters for a cross-backend resume. The file and its
                SHA256 sidecar are required, locally recomputed data statistics
                must agree numerically, and the checkpoint contract must still
                match the exact serialized parameters. Resume-only.
        --- Input arguments specific for hybrid ab initio and empirical models:
            hybrid: whether to train hybrid ab initio and empirical models.
            obs_train_data_path: paths to training data with trajectories with observable values.
            obs_batch_size: training batch size for observable loss in number of frames.
            obs_s_pref: starting prefactor for observable loss.
            obs_l_pref: limit prefactor for observable loss.
            obs_temperature: Temperature of the system (K). Used in the reweighting of observables.
            obs_target: Target value of the observable to be learned. Can be a float or a path to a .npy file containing (single or multiple) values for each configuration in different lines.
            obs_step_every: evaluate and optimize observable loss function every this many steps.
    '''
    
    TIC = time.time()
    if step <= 0:
        raise ValueError('step must be positive.')
    if checkpoint_every is not None and checkpoint_every <= 0:
        raise ValueError('checkpoint_every must be positive.')
    if max_updates_per_run is not None and max_updates_per_run <= 0:
        raise ValueError('max_updates_per_run must be positive.')
    if model_params_path is not None and not resume:
        raise ValueError('model_params_path is only valid when resuming.')
    if model_params_path is not None and model_type != 'energy':
        raise ValueError('model_params_path currently supports energy models only.')
    checkpointing = any((checkpoint_path is not None, checkpoint_every is not None,
                         resume, max_updates_per_run is not None))
    if checkpointing and checkpoint_path is None:
        checkpoint_path = save_path + '.train.pkl'
    if history_path is None:
        history_path = save_path + '.history.json'
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**31 - 1))
    seed = int(seed)
    print('# Training seed:', seed)
    if jax.device_count() > 1:
        print('# Note: Currently only one device will be used for training.')

    # width check
    if fit_widths is None:
        if model_type not in ('atomic', 'atomic_t2'):
            fit_widths = [128, 128, 128]
        else:
            width = embed_mp_widths[-1] if mp else embed_widths[-1]
            fit_widths = [width, width, width]
    for i in range(len(embed_widths)-1):
        if embed_widths[i+1] % embed_widths[i] != 0:
            raise ValueError('embed_widths[i] must divide embed_widths[i+1]')
    if mp:
        if embed_widths[-1] != embed_mp_widths[0]:
            raise ValueError('embed_widths[-1] must equal embed_mp_widths[0].')
        for i in range(len(embed_mp_widths)-1):
            if embed_mp_widths[i+1] % embed_mp_widths[i] != 0 and embed_mp_widths[i+1] % embed_widths[i] != 0:
                raise ValueError('embed_mp_widths[i] must divide or be divisible by embed_mp_widths[i+1]')
    for i in range(len(fit_widths)-1):
        if fit_widths[i+1] != fit_widths[i] != 0:
            print('# Warning: it is recommended to use the same width for all layers in the fitting network.')
    if model_type in ('atomic', 'atomic_t2'):
        if mp:
            if embed_mp_widths[-1] != fit_widths[-1]:
                raise ValueError('For atomic mp models, embed_mp_widths[-1] must equal fit_widths[-1].')
        else:
            if embed_widths[-1] != fit_widths[-1]:
                raise ValueError('For atomic models, embed_widths[-1] must equal fit_widths[-1].')
    assert loss in ('l1-mixed', 'l2'), 'loss must be "l1-mixed" or "l2"'
    # load dataset
    if 'atomic' in model_type and atomic_data_prefix is None:
        atomic_data_prefix = {'atomic':'atomic_dipole', 'atomic_t2':'atomic_polarizability', 'atomic_scalar':'atomic_energy'}[model_type]
    if model_type in ('energy', 'dplr'):
        labels = ['coord', 'box', 'force', 'energy']
    elif 'atomic' in model_type:
        labels = ['coord', 'box', atomic_data_prefix]
        print(f'# Using {atomic_data_prefix}.npy as dataset labels.')
        assert type(atomic_sel) == list, ' Must provide atomic_sel properly for an atomic model.'
    else:
        raise ValueError('model_type should be "energy", "atomic", "atomic_t2", "atomic_scalar", or "dplr".')
    data_params = {'atomic_sel':atomic_sel, 'atomic_scalar':model_type == 'atomic_scalar'}
    dataset_rng = lambda stream: np.random.default_rng(
        np.random.SeedSequence([seed, int(stream)]))
    train_data = Dataset(train_data_path,
                           labels,
                           data_params,
                           rng=dataset_rng(0))
    train_data.compute_lattice_candidate(rcut, use_neighbor_list_when_possible, mp)
    chemical_types = train_data.chemical_types

    # Setup for hybrid training
    if hybrid:
        if model_type != 'energy':
            raise ValueError('For hybrid models model_type has to be energy')
        # Define file names a.k.a. "labels"
        labels_obs = labels + ['observable']
        labels_obs = [item for item in labels_obs if item != 'force']
        # Validate and normalize obs_train_data_path
        if obs_train_data_path is None:
            raise ValueError('Must provide obs_train_data_path for hybrid models.')
        if isinstance(obs_train_data_path, str):
            obs_train_data_path = [obs_train_data_path]
        elif isinstance(obs_train_data_path, list):
            if len(obs_train_data_path) == 0:
                raise ValueError('obs_train_data_path list cannot be empty.')
            # Validate all elements are strings
            for i, path in enumerate(obs_train_data_path):
                if not isinstance(path, str):
                    raise ValueError(f'obs_train_data_path[{i}] must be a string, got {type(path).__name__}')
        else:
            raise ValueError(f'obs_train_data_path must be a string or list of strings, got {type(obs_train_data_path).__name__}')
        # Validate and normalize obs_temperature
        if obs_temperature is None:
            raise ValueError('Must provide obs_temperature for hybrid models.')
        if isinstance(obs_temperature, (int, float)):
            obs_temperature = [obs_temperature]
        elif isinstance(obs_temperature, list):
            if len(obs_temperature) == 0:
                raise ValueError('obs_temperature list cannot be empty.')
        else:
            raise ValueError(f'obs_temperature must be a number or list of numbers, got {type(obs_temperature).__name__}')
        # Validate temperature values
        for i, temp in enumerate(obs_temperature):
            if not isinstance(temp, (int, float)):
                raise ValueError(f'obs_temperature[{i}] must be a number, got {type(temp).__name__}')
            if temp <= 0:
                raise ValueError(f'obs_temperature[{i}] must be positive, got {temp} K')
        # Validate and normalize obs_target
        if obs_target is None:
            raise ValueError('Must provide obs_target for hybrid models')
        if isinstance(obs_target, (int, float)):
            obs_target = [obs_target]
        elif isinstance(obs_target, str):
            try:
                obs_target = [np.load(obs_target)]
            except Exception as e:
                raise ValueError(f'Failed to load obs_target from file {obs_target}: {e}')
        elif isinstance(obs_target, list):
            if len(obs_target) == 0:
                raise ValueError('obs_target list cannot be empty.')
            parsed_obs_target = []
            for i, item in enumerate(obs_target):
                if isinstance(item, str):
                    try:
                        loaded = np.load(item)
                        parsed_obs_target.append(loaded)
                    except Exception as e:
                        raise ValueError(f'Failed to load obs_target[{i}] from file {item}: {e}')
                elif isinstance(item, (int, float)):
                    parsed_obs_target.append(item)
                else:
                    raise ValueError(f'obs_target[{i}] must be a number or path string, got {type(item).__name__}')
            obs_target = parsed_obs_target
        else:
            raise ValueError(f'obs_target must be a number, file path, or list, got {type(obs_target).__name__}')
        # Check consistency of lengths
        n_temps = len(obs_temperature)
        n_targets = len(obs_target)
        n_paths = len(obs_train_data_path)
        if n_temps != n_targets:
            raise ValueError(f'Length mismatch: obs_temperature has {n_temps} entries, but obs_target has {n_targets} entries. They must be equal.')
        if n_temps != n_paths:
            raise ValueError(f'Length mismatch: obs_temperature has {n_temps} entries, but obs_train_data_path has {n_paths} entries. They must be equal.')
        # Load observable data
        train_data_obs = []
        for i in range(n_paths):
            single_data_obs = Dataset(obs_train_data_path[i],
                                        labels_obs,
                                        data_params,
                                        chemical_types=chemical_types,
                                        rng=dataset_rng(100 + i))
            single_data_obs.fill_type(train_data.ntypes)
            single_data_obs.compute_lattice_candidate(rcut, use_neighbor_list_when_possible, mp)
            train_data_obs.append(single_data_obs)

    use_val_data = val_data_path is not None
    if use_val_data:
        val_data = Dataset(val_data_path,
                             labels,
                             data_params,
                             chemical_types=chemical_types,
                             rng=dataset_rng(1))
        val_data.fill_type(train_data.ntypes)
        val_data.compute_lattice_candidate(rcut, use_neighbor_list_when_possible, mp)
    else:
        val_data = None

    # for dplr, convert dataset to short-range
    if model_type == 'dplr':

        if type(dplr_wannier_model_path) is not str:
            raise ValueError('Must properly provide dplr_wannier_model_path (path to your trained Wannier model) for model_type "dplr".')
        if type(dplr_q_atoms) is not list:
            raise ValueError('Must properly provide dplr_q_atoms for model_type "dplr".')
        if type(dplr_q_wc) is not list:
            raise ValueError('Must properly provide dplr_q_wc for model_type "dplr".')
        wc_model, wc_variables = load_model(dplr_wannier_model_path, replicate=False)
        if len(dplr_q_wc) != len(wc_model.params['nsel']):
            raise ValueError('dplr_q_wc must correspond to atomic_sel of the Wannier model.')
        subsets = train_data.get_flattened_data()
        if use_val_data:
            subsets += val_data.get_flattened_data()
        print('# Building short-range dataset...', end='')
        tic_sr = time.time()
        for subset in subsets:
            process_long_range_subset(subset,
                                    dplr_q_atoms,
                                    dplr_q_wc,
                                    dplr_beta,
                                    dplr_resolution,
                                    wc_model,
                                    wc_variables,
                                    use_neighbor_list_when_possible=use_neighbor_list_when_possible)
        print(' Done. Time: %d s' % (time.time() - tic_sr))

    scalar_stats = get_atomic_scalar_stats(train_data, atomic_sel) if model_type == 'atomic_scalar' else None

    # construct model
    params = {
        'type': model_type,
        'atomic_data_prefix': atomic_data_prefix if 'atomic' in model_type else None,
        'embed_widths': embed_widths[:-1] if mp else embed_widths,
        'embedMP_widths': embed_widths[-1:] + embed_mp_widths if mp else None,
        'fit_widths': fit_widths,
        'axis': axis_neurons,
        'Ebias': scalar_stats[0] if model_type == 'atomic_scalar' else (None if 'atomic' in model_type else train_data.fit_energy()),
        'rcut': rcut,
        'use_2nd': True,
        'use_mp': mp,
        'atomic': 'atomic' in model_type,
        'hybrid': hybrid,
        'nsel': atomic_sel if 'atomic' in model_type else None,
        'out_norm': scalar_stats[1] if model_type == 'atomic_scalar' else (train_data.get_atomic_label_scale() if 'atomic' in model_type else 1.),
        **train_data.get_stats(rcut, getstat_bs),
    }
    portable_model_params_sha256 = None
    if model_params_path is not None:
        params, portable_model_params_sha256 = _load_portable_model_params(
            model_params_path, params)
    if model_type == 'dplr':
        dplr_params = {
            'dplr_wannier_model_and_variables': (wc_model, wc_variables),
            'dplr_q_atoms': dplr_q_atoms,
            'dplr_q_wc': dplr_q_wc,
            'dplr_beta': dplr_beta,
            'dplr_resolution': dplr_resolution,
        }
        params.update(dplr_params)
    model = DPModel(params)
    print('# Model params:', {k:v for k,v in model.params.items() if k != 'dplr_wannier_model_and_variables'})

    # initialize model variables
    batch, type_idx, lattice_args = train_data.get_batch(1)
    static_args = _get_static_args(type_idx, lattice_args)
    variables = model.init(
                    jax.random.PRNGKey(seed),
                    batch['coord'][0],
                    batch['box'][0],
                    static_args,
                )
    print('# Model initialized with parameter count %d.' %
           sum(i.size for i in jax.tree_util.tree_flatten(variables)[0]))
    
    # initialize optimizer
    if lr is None:
        lr = 0.002 if 'atomic' not in model_type else 0.01
    if step < decay_steps * 10:
        decay_steps = max(step // 10, 1)
    lr_scheduler = optax.exponential_decay(
                        init_value = lr,
                        transition_steps = decay_steps,
                        decay_rate = (lr_limit/lr) ** (decay_steps / (step-decay_steps)),
                        transition_begin = 0,
                        staircase = True,
                    )
    optimizer = optax.adam(learning_rate = lr_scheduler,
                           b2 = beta2)
    opt_state = optimizer.init(variables)

    # define training step
    loss_fn, loss_and_grad_fn = model.get_loss_fn(order=loss)
    print(f'# Using {loss} loss function.')
    if hybrid:
        loss_obs, loss_and_grad_obs = model.get_observable_loss_fn()
    print('# Optimizer initialized with initial lr = %.1e. Starting training...' % lr)

    if 'atomic' not in model_type:
        state = {'loss_avg': 0., 'le_avg': 0., 'lf_avg': 0., 'iteration': 0}
    else:
        state = {'loss_avg': 0., 'iteration': 0}
    if hybrid:
        state_obs = {
            k: {'lobs_avg': 0., 'obs_term_avg': 0., 'obs_mean': 0.,
                'logweights': [0.], 'ESS': 1.}
            for k in range(len(obs_train_data_path))
        }

    model_params_raw = pickle.dumps(_tree_to_host(model.params),
                                    protocol=pickle.HIGHEST_PROTOCOL)
    model_params_sha256 = (portable_model_params_sha256
                           if portable_model_params_sha256 is not None
                           else hashlib.sha256(model_params_raw).hexdigest())
    contract = {
        'model_type': model_type,
        'rcut': rcut,
        'train_data_path': _path_fingerprint(train_data_path),
        'val_data_path': _path_fingerprint(val_data_path),
        'step': step,
        'mp': mp,
        'atomic_sel': atomic_sel,
        'embed_widths': embed_widths,
        'embed_mp_widths': embed_mp_widths,
        'fit_widths': fit_widths,
        'axis_neurons': axis_neurons,
        'lr': lr,
        'lr_limit': lr_limit,
        'beta2': beta2,
        'decay_steps': decay_steps,
        'batch_size': batch_size,
        'val_batch_size_ratio': val_batch_size_ratio,
        'print_every': print_every,
        'label_bs': label_bs,
        'loss': loss,
        's_pref_e': s_pref_e,
        'l_pref_e': l_pref_e,
        's_pref_f': s_pref_f,
        'l_pref_f': l_pref_f,
        'seed': seed,
        'hybrid': hybrid,
        'obs_train_data_path': _path_fingerprint(obs_train_data_path),
        'obs_batch_size': obs_batch_size,
        'obs_temperature': obs_temperature,
        'obs_target': obs_target,
        'obs_step_every': obs_step_every,
        'use_neighbor_list_when_possible': use_neighbor_list_when_possible,
        # A portable file is the content-addressed representation accepted by
        # the source checkpoint. Re-pickling its loaded tree is not a stable
        # identity operation across NumPy/JAX backends, even when every value
        # and dtype is unchanged, so retain the already verified file digest.
        'model_params_sha256': model_params_sha256,
    }
    contract_sha256 = hashlib.sha256(
        pickle.dumps(contract, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    history = []

    if resume:
        if checkpoint_path is None or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError('Training checkpoint not found: %s' % checkpoint_path)
        checkpoint = _load_training_checkpoint(checkpoint_path)
        if checkpoint.get('contract_sha256') != contract_sha256:
            checkpoint_contract = checkpoint.get('contract') or {}
            differing_fields = sorted(
                key for key in set(checkpoint_contract) | set(contract)
                if pickle.dumps(checkpoint_contract.get(key),
                                protocol=pickle.HIGHEST_PROTOCOL)
                != pickle.dumps(contract.get(key),
                                protocol=pickle.HIGHEST_PROTOCOL))
            raise ValueError(
                'Training checkpoint contract does not match this run: '
                'checkpoint_sha256=%s run_sha256=%s differing_fields=%s.' %
                (checkpoint.get('contract_sha256'), contract_sha256,
                 ','.join(differing_fields) or '<unavailable>'))
        variables = checkpoint['variables']
        opt_state = checkpoint['opt_state']
        state = checkpoint['state']
        history = checkpoint.get('history', [])
        train_data.set_sampler_state(checkpoint['train_sampler_state'])
        if use_val_data:
            val_data.set_sampler_state(checkpoint['val_sampler_state'])
        if hybrid:
            state_obs = checkpoint['state_obs']
            for dataset, sampler_state in zip(train_data_obs,
                                              checkpoint['obs_sampler_states']):
                dataset.set_sampler_state(sampler_state)
        print('# Resumed exact training state at update %d from \'%s\'.' %
              (int(np.asarray(state['iteration'])), checkpoint_path))

    def save_training_state():
        if not checkpointing:
            return
        payload = {
            'checkpoint_version': _CHECKPOINT_VERSION,
            'contract_sha256': contract_sha256,
            'contract': contract,
            'variables': _tree_to_host(variables),
            'opt_state': _tree_to_host(opt_state),
            'state': _tree_to_host(state),
            'history': history,
            'train_sampler_state': train_data.get_sampler_state(),
            'val_sampler_state': (val_data.get_sampler_state()
                                  if use_val_data else None),
            'state_obs': _tree_to_host(state_obs) if hybrid else None,
            'obs_sampler_states': ([dataset.get_sampler_state()
                                    for dataset in train_data_obs]
                                   if hybrid else None),
        }
        _save_training_checkpoint(checkpoint_path, payload)
        _write_history(history_path, history)
        print('# Training checkpoint saved at update %d to \'%s\'.' %
              (int(np.asarray(state['iteration'])), checkpoint_path))

    @partial(jax.jit, static_argnames=('static_args',))
    def train_step(batch, variables, opt_state, state, static_args):
        r = lr_scheduler(state['iteration']) / lr
        if 'atomic' not in model_type:
            pref = {'e': s_pref_e*r + l_pref_e*(1-r),
                    'f': s_pref_f*r + l_pref_f*(1-r)}
            (loss_total, (loss_e, loss_f)), grads = loss_and_grad_fn(variables,
                                                                    batch,
                                                                    pref,
                                                                    static_args)
            for key, value in zip(['loss_avg', 'le_avg', 'lf_avg'],
                                  [loss_total, loss_e, loss_f]):
                state[key] = state[key] * (1-1/print_loss_smoothing) + value
        else:
            loss_total, grads = loss_and_grad_fn(variables,
                                                 batch,
                                                 static_args)
            state['loss_avg'] = state['loss_avg'] * (1-1/print_loss_smoothing) + loss_total
        updates, opt_state = optimizer.update(grads, opt_state, variables)
        variables = optax.apply_updates(variables, updates)
        state['iteration'] += 1
        return variables, opt_state, state
    
    @partial(jax.jit, static_argnames=('static_args', 'obs_position'))
    def train_step_obs(batch, variables, opt_state, state_obs, static_args, obs_position=0):
        r = lr_scheduler(state['iteration']) / lr
        pref = {'obs': obs_s_pref*r + obs_l_pref*(1-r)}
        (loss_obs, (loss_obs_raw, obs_avg, obs_batch, logweights)), grads = loss_and_grad_obs(variables,
                                                                                            batch,
                                                                                            pref,
                                                                                            static_args,
                                                                                            obs_temperature[obs_position],
                                                                                            obs_target[obs_position])
        state_obs[obs_position]['lobs_avg'] = state_obs[obs_position]['lobs_avg'] * (1-1/print_loss_smoothing) + jnp.sqrt(loss_obs_raw) * 1/print_loss_smoothing
        state_obs[obs_position]['obs_term_avg'] = obs_avg
        state_obs[obs_position]['obs_mean'] = np.mean(obs_batch, axis=0)
        state_obs[obs_position]['logweights'] = logweights
        weights = jnp.exp(logweights)
        state_obs[obs_position]['ESS'] = jnp.sum(weights)**2 / jnp.sum(weights ** 2)
        updates, opt_state = optimizer.update(grads, opt_state, variables)
        variables = optax.apply_updates(variables, updates)
        return variables, opt_state, state_obs
    
    # define validation step
    @partial(jax.jit, static_argnames=('static_args',))
    def val_step(batch, variables, static_args):
        if 'atomic' not in model_type:
            pref = {'e': 1, 'f': 1}
            _, (loss_e, loss_f) = loss_fn(variables,
                                          batch,
                                          pref,
                                          static_args)
            return loss_e, loss_f
        else:
            loss_total = loss_fn(variables,
                                 batch,
                                 static_args)
            return loss_total
        
    # configure batch size
    if batch_size is None:
        print(f'# Auto batch size = int({label_bs}/nlabels_per_frame)')
    else:
        print(f'# Batch size = {batch_size}')
    if hybrid:
        print(f'# Observable loss batch size = {obs_batch_size}')
    def get_batch_train():
        if batch_size is None:
            return train_data.get_batch(label_bs, 'label')
        else:
            return train_data.get_batch(batch_size)
    def get_batch_train_obs(obs_position=0):
        return train_data_obs[obs_position].get_batch(obs_batch_size)
    def get_batch_val():
        ret = []
        for _ in range(val_batch_size_ratio):
            if batch_size is None:
                ret.append(val_data.get_batch(label_bs, 'label'))
            else:
                ret.append(val_data.get_batch(batch_size))
        return ret
        
    # define print step
    def print_step(loss_val, elapsed):
        completed = int(np.asarray(state['iteration']))
        beta_smoothing = print_loss_smoothing * (
            1 - (1 - 1 / print_loss_smoothing) ** completed)
        line = f'Update {completed:7d}'
        record = {'update': completed,
                  'learning_rate': float(np.asarray(lr_scheduler(completed - 1)))}
        L_train = float(np.asarray(state["loss_avg"] / beta_smoothing))
        L_print = L_train if loss == 'l1-mixed' else L_train ** 0.5
        record['loss'] = L_print
        line += f' L {L_print:7.5f}'
        if 'atomic' not in model_type:
            LE_train = float(np.asarray(state["le_avg"] / beta_smoothing))
            LF_train = float(np.asarray(state["lf_avg"] / beta_smoothing))
            LE_print = LE_train if loss == 'l1-mixed' else LE_train ** 0.5
            LF_print = LF_train if loss == 'l1-mixed' else LF_train ** 0.5
            record.update(energy_loss=LE_print, force_loss=LF_print)
            line += f' LE {LE_print:7.5f}'
            line += f' LF {LF_print:7.5f}'
        if hybrid:
            for obs_position in range(len(obs_train_data_path)):
                lobs = float(state_obs[obs_position]["lobs_avg"])
                ess = float(state_obs[obs_position]["ESS"])
                record[f'observable_loss_{obs_position}'] = lobs
                record[f'observable_ess_{obs_position}'] = ess
                line += f' LOBS{obs_position} {lobs:7.5f}'
                line += f' ESS{obs_position} {ess:7.5f}'
                for obs_item  in range(len(state_obs[obs_position]["obs_term_avg"])):
                    obs_rew = float(state_obs[obs_position]["obs_term_avg"][obs_item])
                    obs_mean = float(state_obs[obs_position]["obs_mean"][obs_item])
                    record[f'observable_reweighted_{obs_position}_{obs_item}'] = obs_rew
                    record[f'observable_mean_{obs_position}_{obs_item}'] = obs_mean
                    line += f' OBS_REW_{obs_position}_{obs_item} {obs_rew:7.5f}'
                    line += f' OBS_{obs_position}_{obs_item} {obs_mean:7.5f}'
        if use_val_data:
            if 'atomic' not in model_type:
                LEval = float(np.array([l[0] for l in loss_val]).mean())
                LFval = float(np.array([l[1] for l in loss_val]).mean())
                LEval_print = LEval if loss == 'l1-mixed' else LEval ** 0.5
                LFval_print = LFval if loss == 'l1-mixed' else LFval ** 0.5
                record.update(validation_energy_loss=LEval_print,
                              validation_force_loss=LFval_print)
                line += f' LEval {LEval_print:7.5f}'
                line += f' LFval {LFval_print:7.5f}'
            else:
                Lval = float(np.array(loss_val).mean())
                Lval_print = Lval if loss == 'l1-mixed' else Lval ** 0.5
                record['validation_loss'] = Lval_print
                line += f' Lval {Lval_print:7.5f}'
        line += f' Time {elapsed:.2f}s'
        print(line)
        history.append(record)

    # training loop
    tic = time.time()
    segment_start = int(np.asarray(state['iteration']))
    while int(np.asarray(state['iteration'])) < step:
        iteration = int(np.asarray(state['iteration']))
        batch, type_idx, lattice_args = get_batch_train()
        static_args = _get_static_args(type_idx, lattice_args)
        variables, opt_state, state = train_step(batch,
                                                 variables,
                                                 opt_state,
                                                 state,
                                                 static_args)
        
        # training step part 2 in hybrid observable training
        if hybrid and iteration % obs_step_every == 0:
            # observable train step
            for i in range(len(obs_train_data_path)):
                batch, type_idx, lattice_args = get_batch_train_obs(obs_position=i)
                static_args = _get_static_args(type_idx, lattice_args)
                variables, opt_state, state_obs = train_step_obs(batch,
                                                        variables,
                                                        opt_state,
                                                        state_obs,
                                                        static_args,
                                                        obs_position=i) 

        completed = int(np.asarray(state['iteration']))
        report_this_update = completed % print_every == 0 or completed == step
        if report_this_update:
            loss_val = None
            if use_val_data:
                val_batch = get_batch_val()
                loss_val = []
                for one_batch in val_batch:
                    v_batch, type_idx, lattice_args = one_batch
                    static_args = _get_static_args(type_idx, lattice_args)
                    loss_val.append(val_step(v_batch, variables, static_args))
            print_step(loss_val, time.time() - tic)
            tic = time.time()

        if checkpoint_every is not None and completed % checkpoint_every == 0:
            save_training_state()
        if (max_updates_per_run is not None
                and completed - segment_start >= max_updates_per_run
                and completed < step):
            save_training_state()
            print('# Training segment stopped cleanly at update %d/%d.' %
                  (completed, step))
            return {'completed': False, 'completed_updates': completed,
                    'target_updates': step, 'checkpoint_path': checkpoint_path}

    # compress, save, and finish
    save_training_state()
    if compress:
        model, variables = compress_model(model,
                                                variables,
                                                compress_Ngrids,
                                                compress_r_min)
    save_model(save_path, model, variables)
    print(f'# Training finished in {datetime.timedelta(seconds=int(time.time() - TIC))}.')
    return {'completed': True, 'completed_updates': int(np.asarray(state['iteration'])),
            'target_updates': step, 'model_path': save_path,
            'checkpoint_path': checkpoint_path if checkpointing else None}


def test(
    model_path: str,
    data_path: Union[str, List[str]],
    batch_size: int = 1,
):
    '''
        Testing a trained model on one or more datasets.
        Input arguments:
            model_path: path to the trained model.
            data_path: path, or list of paths, to the data for evaluation.
            batch_size: Increase for potentially faster evaluation, but requires more memory.
    '''
    if jax.device_count() > 1:
        print('# Note: Currently only one device will be used for evaluation.')

    model, variables = load_model(model_path, replicate=False)
    if model.params['type'] in ('energy', 'dplr'):
        labels = ['coord', 'box', 'force', 'energy']
        atomic_sel = None
    elif 'atomic' in model.params['type']:
        labels = ['coord', 'box', model.params['atomic_data_prefix']]
        atomic_sel = model.params['nsel']
    else:
        raise ValueError('Model type should be "energy", "atomic", "atomic_t2", "atomic_scalar", or "dplr".')
    test_data = Dataset(data_path,
                        labels,
                        {'atomic_sel': atomic_sel, 'atomic_scalar': model.params['type'] == 'atomic_scalar'},
                        chemical_types=model.params.get('chemical_types'))
    test_data.fill_type(model.params['ntypes'])
    test_data.compute_lattice_candidate(model.params['rcut'],
                                        mp=model.params.get('use_mp', False))
    if 'dplr' in model.params['type']:
        subsets = test_data.get_flattened_data()
        for subset in subsets:
            process_long_range_subset(subset,
                                      model.params['dplr_q_atoms'],
                                      model.params['dplr_q_wc'],
                                      model.params['dplr_beta'],
                                      model.params['dplr_resolution'],
                                      *model.params['dplr_wannier_model_and_variables'],
                                      keep_long_range=True)

    if model.params['type'] in ('energy', 'dplr'):
        evaluate_fn = model.energy_and_force
        stats = {
            'energy': {'sq': 0.0, 'abs': 0.0, 'count': 0},
            'force': {'sq': 0.0, 'abs': 0.0, 'count': 0},
            'force_l1_sum': 0.0,
            'force_l1_count': 0,
            'energy_l1_sum': 0.0,
            'energy_l1_count': 0,
        }
    else:
        key = model.params['atomic_data_prefix']
        evaluate_fn = lambda variables, coord, box, static_args: model.apply(variables, coord, box, static_args)
        stats = {
            key: {'sq': 0.0, 'abs': 0.0, 'count': 0},
            'l1_sum': 0.0,
            'l1_count': 0,
        }
    test_results = []

    evaluate_fn = jax.jit(
        jax.vmap(evaluate_fn, in_axes=(None, 0, 0, None)),
        static_argnames=('static_args',)
    )

    for leaf in test_data.get_leaves():
        leaf.pointer = 0
        remaining = leaf.nframes
        while remaining > 0:
            bs = min(batch_size, remaining)
            batch, type_idx, lattice_args = leaf.get_batch(bs)
            remaining -= bs

            static_args = _get_static_args(type_idx, lattice_args)
            pred = evaluate_fn(variables, batch['coord'], batch['box'], static_args)
            source_index = batch.get('_source_index')

            if model.params['type'] in ('energy', 'dplr'):
                E_pred, F_pred = pred
                E_true = batch['energy']
                F_true = batch['force']
                E_lr = F_lr = None
                if model.params['type'] == 'dplr':
                    E_lr = batch['_dplr_long_range_energy']
                    F_lr = batch['_dplr_long_range_force']
                    E_pred = E_pred + E_lr
                    F_pred = F_pred + F_lr
                    E_true = E_true + E_lr
                    F_true = F_true + F_lr

                type_idx_arr = np.asarray(type_idx, dtype=int)
                for i in range(E_pred.shape[0]):
                    result = {
                        'box': np.asarray(batch['box'][i]).copy(),
                        'type_idx': type_idx_arr.copy(),
                        'predicted_energy': float(np.asarray(E_pred[i])),
                        'true_energy': float(np.asarray(E_true[i])),
                        'predicted_force': np.asarray(F_pred[i]).copy(),
                        'true_force': np.asarray(F_true[i]).copy(),
                    }
                    if model.params['type'] == 'dplr':
                        result['long_range_energy'] = float(np.asarray(E_lr[i]))
                        result['long_range_force'] = np.asarray(F_lr[i]).copy()
                    if source_index is not None:
                        result['_source_index'] = int(np.asarray(source_index[i]))
                    test_results.append(result)

                natoms = F_pred.shape[1]
                dE = (E_pred - E_true) / natoms
                stats['energy']['sq'] += (dE**2).sum()
                stats['energy']['abs'] += np.abs(dE).sum()
                stats['energy']['count'] += dE.size
                stats['energy_l1_sum'] += np.abs(dE).sum()
                stats['energy_l1_count'] += dE.size

                diffF = F_pred - F_true
                stats['force']['sq'] += (diffF**2).sum()
                stats['force']['abs'] += np.abs(diffF).sum()
                stats['force']['count'] += diffF.size
                per_atom = (diffF**2).mean(-1)**0.5
                stats['force_l1_sum'] += per_atom.sum()
                stats['force_l1_count'] += per_atom.size
            else:
                key = model.params['atomic_data_prefix']
                pred_val = pred[0]
                true_val = batch['atomic']

                type_idx_arr = np.asarray(type_idx, dtype=int)
                for i in range(pred_val.shape[0]):
                    result = {
                        'box': np.asarray(batch['box'][i]).copy(),
                        'type_idx': type_idx_arr.copy(),
                        'atomic_data_prefix': key,
                        'predicted_atomic': np.asarray(pred_val[i]).copy(),
                        'true_atomic': np.asarray(true_val[i]).copy(),
                    }
                    if source_index is not None:
                        result['_source_index'] = int(np.asarray(source_index[i]))
                    test_results.append(result)

                diff = pred_val - true_val
                stats[key]['sq'] += (diff**2).sum()
                stats[key]['abs'] += np.abs(diff).sum()
                stats[key]['count'] += diff.size
                per_atom = (diff**2).mean(tuple(range(2,diff.ndim)))**0.5
                stats['l1_sum'] += per_atom.sum()
                stats['l1_count'] += per_atom.size

    rmse = {}
    mae = {}
    l1_mixed = {}

    if model.params['type'] in ('energy', 'dplr'):
        rmse['energy'] = (stats['energy']['sq'] / stats['energy']['count'])**0.5
        mae['energy'] = stats['energy']['abs'] / stats['energy']['count']
        rmse['force'] = (stats['force']['sq'] / stats['force']['count'])**0.5
        mae['force'] = stats['force']['abs'] / stats['force']['count']
        l1_mixed['energy'] = stats['energy_l1_sum'] / stats['energy_l1_count']
        l1_mixed['force'] = stats['force_l1_sum'] / stats['force_l1_count']
    else:
        key = model.params['atomic_data_prefix']
        rmse[key] = (stats[key]['sq'] / stats[key]['count'])**0.5
        mae[key] = stats[key]['abs'] / stats[key]['count']
        l1_mixed[key] = stats['l1_sum'] / stats['l1_count']

    if test_results and '_source_index' in test_results[0]:
        test_results.sort(key=lambda result: result['_source_index'])
        for result in test_results:
            del result['_source_index']

    def _metric_dict_to_float(metric):
        return {key: float(np.asarray(value).item()) for key, value in metric.items()}

    error_metrics = {
        'rmse': _metric_dict_to_float(rmse),
        'mae': _metric_dict_to_float(mae),
        'l1_mixed': _metric_dict_to_float(l1_mixed),
    }
    return error_metrics, test_results

def evaluate(
    model_path: str,
    coord: np.ndarray,
    box: np.ndarray,
    type_idx: np.ndarray,
    batch_size: int = 1,
):
    '''
        Evaluating a trained model on a set of configurations (without knowing ground truth).
        Input arguments:
            model_path: path to the trained model.
            coord: atomic coordinates of shape (n_frames, n_atoms, 3).
            box: simulation box of shape (n_frames) + (,) or (1,) or (3,) or (9), or (3,3).
            type_idx: atomic type indices of shape (Natoms,). If the model was trained with
                chemical_types (extxyz path), type_idx is interpreted as atomic numbers (Z).
            batch_size: Increase for potentially faster evaluation, but requires more memory.
    '''
    # input shape check
    try:
        assert coord.ndim == 3 and coord.shape[2] == 3
        assert type_idx.ndim == 1 and box.ndim in [1, 2, 3]
        assert coord.shape[1] == type_idx.shape[0]
        assert coord.shape[0] == box.shape[0]
        if box.ndim == 1:
            box = box[:, None, None] * jnp.eye(3)
        elif box.ndim == 2:
            if box.shape[1] == 1:
                box = box[:, None] * jnp.eye(3)
            elif box.shape[1] == 3:
                box = jax.vmap(jnp.diag)(box)
            else:
                box = box.reshape(box.shape[0], 3, 3)
        elif box.ndim == 3:
            assert box.shape[1] == 3 and box.shape[2] == 3
    except:
        raise ValueError('Input shapes are incorrect: \n' + 
                         'coord: (n_frames, n_atoms, 3) \n' +
                         'box: (n_frames) + (,) or (1,) or (3,) or (9), or (3,3) \n' +
                         'type_idx (n_atoms).')
    
    model, _ = load_model(model_path, replicate=False)
    ct = model.params.get('chemical_types')
    if ct is not None:
        type_idx_np = np.array(type_idx, dtype=int)
        unknown = set(type_idx_np.tolist()) - set(ct)
        if unknown:
            raise ValueError('Atomic numbers %s in type_idx are not in model.params["chemical_types"]=%s'
                             % (sorted(unknown), ct))
        z_to_idx = {z: i for i, z in enumerate(ct)}
        type_idx = np.array([z_to_idx[z] for z in type_idx_np], dtype=int)

    # create dataset in temp directory and use test() to evaluate
    with tempfile.TemporaryDirectory() as temp_dir:
        set_dir = os.path.join(temp_dir, "set.001")
        coord_path = os.path.join(set_dir, "coord.npy")
        box_path = os.path.join(set_dir, "box.npy")
        type_idx_path = os.path.join(temp_dir, "type.raw")
        os.makedirs(set_dir, exist_ok=True)
        np.save(coord_path, coord.reshape(coord.shape[0], -1))
        np.save(box_path, box.reshape(box.shape[0], -1))
        with open(type_idx_path, "w") as f:
            f.write("\n".join(np.array(type_idx, dtype=int).astype(str)))
        if 'atomic' in model.params['type']:
            atomic_path = os.path.join(set_dir, model.params['atomic_data_prefix'] + ".npy")
            label_count = np.isin(type_idx, model.params['nsel']).sum()
            label_dim = {'atomic_t2':9, 'atomic':3, 'atomic_scalar':1}[model.params['type']]
            np.save(atomic_path, np.zeros((coord.shape[0], label_count * label_dim)))
        elif model.params['type'] in ('energy', 'dplr'):
            energy_path = os.path.join(set_dir, "energy.npy")
            force_path = os.path.join(set_dir, "force.npy")
            np.save(energy_path, np.zeros(coord.shape[0]))
            np.save(force_path, np.zeros((coord.reshape(coord.shape[0], -1)).shape))
        _, test_results = test(model_path, temp_dir, batch_size)

    if model.params['type'] in ('energy', 'dplr'):
        return {
            'energy': np.asarray([
                result['predicted_energy'] for result in test_results]),
            'force': np.stack([
                result['predicted_force'] for result in test_results], axis=0),
        }
    key = model.params['atomic_data_prefix']
    return {
        key: np.stack([
            result['predicted_atomic'] for result in test_results], axis=0)
    }
    
def process_long_range_subset(subset, dplr_q_atoms, dplr_q_wc, dplr_beta, dplr_resolution,
                              wc_model, wc_variables, keep_long_range=False,
                              use_neighbor_list_when_possible=True):
    '''
        subtracting long range energy and force, keeping short range part only, for dplr models.
    '''
    data, type_idx, _ = subset.values()
    lattice_args = compute_lattice_candidate(data['box'], wc_model.params['rcut'])
    if not lattice_args['ortho']:
        raise ValueError('For "dplr" currently only orthorhombic boxes are supported.')
    type_idx = np.asarray(type_idx)
    use_neighborlist = bool(use_neighbor_list_when_possible and
                            len(lattice_args['lattice_cand']) == 1)
    type_count = tuple(np.bincount(type_idx, minlength=wc_model.params['ntypes']))
    max_nbrs = tuple(map(int, np.asarray(jax.jit(lambda coord, box:
        get_max_nbrs(coord, box, tuple(type_idx), type_count,
                     wc_model.params['rcut'], lattice_args['ortho']))(
                         data['coord'], data['box'])))) if use_neighborlist else None
    if max_nbrs is not None and not neighborlist_is_efficient(
            max_nbrs, len(type_idx), wc_model.params.get('use_mp', False)):
        use_neighborlist, max_nbrs = False, None
    lattice_args.update({'use_neighborlist': use_neighborlist,
                         'max_nbrs': max_nbrs})
    qatoms, qwc = dplr_charges(type_idx, dplr_q_atoms, dplr_q_wc,
                               wc_model.params['nsel'], wc_model.params['ntypes'])
    static_args = _get_static_args(type_idx, lattice_args)

    def lr_energy(coord, box, Ngrid):
        wc = wc_model.wc_predict(wc_variables, coord, box, static_args)
        p3mlr_fn = get_p3mlr_fn(jnp.diag(box), dplr_beta, Ngrid)
        return p3mlr_fn(jnp.concatenate([coord, wc]), jnp.concatenate([qatoms, qwc]), jnp.diag(box))
    
    @partial(jax.jit, static_argnums=(2,))
    def lr_energy_and_force(coord, box, Ngrid):
        e, negf = jax.value_and_grad(lr_energy)(coord, box, Ngrid)
        return e, -negf

    if keep_long_range:
        lr_energy_frames = []
        lr_force_frames = []
    for i in range(len(data['coord'])):
        Ngrid = get_p3mlr_grid_size(np.diag(data['box'][i]), dplr_beta, resolution=dplr_resolution)
        e_lr, f_lr = lr_energy_and_force(data['coord'][i], data['box'][i], Ngrid)
        if keep_long_range:
            lr_energy_frames.append(np.asarray(e_lr))
            lr_force_frames.append(np.asarray(f_lr))
        data['energy'][i] -= e_lr
        data['force'][i] -= f_lr
    if keep_long_range:
        data['_dplr_long_range_energy'] = np.asarray(
            lr_energy_frames, dtype=data['energy'].dtype).reshape(data['energy'].shape)
        data['_dplr_long_range_force'] = np.asarray(
            lr_force_frames, dtype=data['force'].dtype).reshape(data['force'].shape)
