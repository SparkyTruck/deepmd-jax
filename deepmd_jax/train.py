import jax
import optax
import numpy as np
import jax.numpy as jnp
import time, datetime
import flax.linen as nn
from functools import partial
from .utils import get_p3mlr_fn, get_p3mlr_grid_size, load_model, save_model, compress_model, save_dataset
from .data import DPDataset
from .dpmodel import DPModel
from typing import Union, List
import tempfile
import os

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
):
    '''
        Entry point for training deepmd-jax models.

        Input arguments:
            model_type:
                 'energy' (standard force field),
                 'atomic' (predicts per-atom 3-vectors, e.g. Wannier centroid),
                 'atomic_t2' (predicts per-atom symmetric 3x3 tensors, e.g. polarizability),
                 'dplr' (force field w/ long-range electrostatics).
            rcut: cutoff radius (Angstrom) for the model.
            save_path: path to save the trained model.
            train_data: path to training data (str) or list of paths to training data (List[str]).
            val_data: path to validation data (str) or list of paths to validation data (List[str]).
            step: number of training steps. Depending on dataset size, expect 1e5-1e7 for energy models and 1e5-1e6 for wannier models.
            mp: whether to use message passing model for more accuracy at a higher cost.
            atomic_sel: Selects the atom types for prediction. Only used when model_type == 'atomic' or 'atomic_t2'.
            embed_widths: Widths of the embedding neural network.
            embed_mp_widths: Widths of the embedding neural network in message passing. Only used when mp == True.
            fit_widths: Widths of the fitting neural network.
            axis_neurons: Number of axis neurons to project the atomic features before the fitting network. Recommended range: 8-16.
            lr: learning rate at start. If None, default values (0.002 for 'energy' and 0.01 for 'atomic'/'atomic_t2') is used.
            batch_size: training batch size in number of frames. If None, will be automatically determined by label_bs.
            val_batch_size_ratio: validation batch size / batch_size. Increase for stabler validation loss.
            compress: whether to compress the model after training for faster inference.
            print_every: interval for printing loss and validation.
            atomic_data_prefix: prefix for .npy label files. Defaults to 'atomic_dipole' for 'atomic' and 'atomic_polarizability' for 'atomic_t2'.
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
    if jax.device_count() > 1:
        print('# Note: Currently only one device will be used for training.')

    # width check
    if fit_widths is None:
        if 'atomic' not in model_type:
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
    if 'atomic' in model_type:
        if mp:
            if embed_mp_widths[-1] != fit_widths[-1]:
                raise ValueError('For atomic mp models, embed_mp_widths[-1] must equal fit_widths[-1].')
        else:
            if embed_widths[-1] != fit_widths[-1]:
                raise ValueError('For atomic models, embed_widths[-1] must equal fit_widths[-1].')
    assert loss in ('l1-mixed', 'l2'), 'loss must be "l1-mixed" or "l2"'
    print(f'# Using {loss} loss function.')

    # load dataset
    if 'atomic' in model_type and atomic_data_prefix is None:
        atomic_data_prefix = 'atomic_dipole' if model_type == 'atomic' else 'atomic_polarizability'
    if model_type == 'energy' or model_type == 'dplr':
        labels = ['coord', 'box', 'force', 'energy']
    elif 'atomic' in model_type:
        labels = ['coord', 'box', atomic_data_prefix]
        print(f'# Using {atomic_data_prefix}.npy as dataset labels.')
        assert type(atomic_sel) == list, ' Must provide atomic_sel properly for model_type "atomic"/"atomic_t2".'
    else:
        raise ValueError('model_type should be "energy", "atomic", "atomic_t2", or "dplr".')
    if type(train_data_path) == str:
        train_data_path = [train_data_path]
    else:
        train_data_path = [[path] for path in train_data_path]
    train_data = DPDataset(train_data_path,
                           labels,
                           {'atomic_sel':atomic_sel})
    train_data.compute_lattice_candidate(rcut)

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
            single_data_obs = DPDataset([obs_train_data_path[i]],
                                        labels_obs,
                                        {'atomic_sel':atomic_sel})
            single_data_obs.compute_lattice_candidate(rcut)
            train_data_obs.append(single_data_obs)

    use_val_data = val_data_path is not None
    if use_val_data:
        if type(val_data_path) == str:
            val_data_path = [val_data_path]
        else:
            val_data_path = [[path] for path in val_data_path]
        val_data = DPDataset(val_data_path,
                             labels,
                             {'atomic_sel':atomic_sel})
        val_data.compute_lattice_candidate(rcut)
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
                                    wc_variables)
        print(' Done. Time: %d s' % (time.time() - tic_sr))

    # construct model
    params = {
        'type': model_type,
        'atomic_data_prefix': atomic_data_prefix if 'atomic' in model_type else None,
        'embed_widths': embed_widths[:-1] if mp else embed_widths,
        'embedMP_widths': embed_widths[-1:] + embed_mp_widths if mp else None,
        'fit_widths': fit_widths,
        'axis': axis_neurons,
        'Ebias': None if 'atomic' in model_type else train_data.fit_energy(),
        'rcut': rcut,
        'use_2nd': True,
        'use_mp': mp,
        'atomic': 'atomic' in model_type,
        'hybrid': hybrid,
        'nsel': atomic_sel if 'atomic' in model_type else None,
        'out_norm': train_data.get_atomic_label_scale() if 'atomic' in model_type else 1.,
        **train_data.get_stats(rcut, getstat_bs),
    }
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
    batch, type_count, lattice_args = train_data.get_batch(1)
    static_args = nn.FrozenDict({'type_count': type_count, 'lattice': lattice_args})
    if seed is None:
        seed = np.random.randint(65536)
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
    print('# Optimizer initialized with initial lr = %.1e. Starting training...' % lr)

    # define training step
    loss_fn, loss_and_grad_fn = model.get_loss_fn(order=loss)
    if hybrid:
        loss_obs, loss_and_grad_obs = model.get_observable_loss_fn()

    if 'atomic' not in model_type:
        state = {'loss_avg': 0., 'le_avg': 0., 'lf_avg': 0., 'iteration': 0}
    else:
        state = {'loss_avg': 0., 'iteration': 0}
    if hybrid:
        state_obs = {}
        _single_state_obs = {'lobs_avg': 0., 'obs_term_avg': 0., 'obs_mean': 0., 'logweights': [0.], 'ESS': 1. }
        state_obs = {k: _single_state_obs for k in range(len(obs_train_data_path))}

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
    def print_step():
        beta_smoothing = print_loss_smoothing * (1 - (1-1/print_loss_smoothing)**(iteration+1))
        line = f'Iter {iteration:7d}'
        L_train = state["loss_avg"] / beta_smoothing
        line += f' L {L_train if loss == "l1-mixed" else L_train ** 0.5:7.5f}'
        if 'atomic' not in model_type:
            LE_train = state["le_avg"] / beta_smoothing
            LF_train = state["lf_avg"] / beta_smoothing
            line += f' LE {LE_train if loss == "l1-mixed" else LE_train ** 0.5:7.5f}'
            line += f' LF {LF_train if loss == "l1-mixed" else LF_train ** 0.5:7.5f}'
        if hybrid:
            for obs_position in range(len(obs_train_data_path)):
                line += f' LOBS{obs_position} {float(state_obs[obs_position]["lobs_avg"]):7.5f}'
                line += f' ESS{obs_position} {float(state_obs[obs_position]["ESS"]):7.5f}'
                for obs_item  in range(len(state_obs[obs_position]["obs_term_avg"])):
                    line += f' OBS_REW_{obs_position}_{obs_item} {float(state_obs[obs_position]["obs_term_avg"][obs_item]):7.5f}'
                    line += f' OBS_{obs_position}_{obs_item} {float(state_obs[obs_position]["obs_mean"][obs_item]):7.5f}'
        if use_val_data:
            if 'atomic' not in model_type:
                LEval = np.array([l[0] for l in loss_val]).mean()
                LFval = np.array([l[1] for l in loss_val]).mean()
                line += f' LEval {LEval if loss == "l1-mixed" else LEval ** 0.5:7.5f}'
                line += f' LFval {LFval if loss == "l1-mixed" else LFval ** 0.5:7.5f}'
            else:
                Lval = np.array(loss_val).mean()
                line += f' Lval {Lval if loss == "l1-mixed" else Lval ** 0.5:7.5f}'
        line += f' Time {time.time() - tic:.2f}s'
        print(line)

    # training loop
    tic = time.time()
    for iteration in range(int(step+1)):
        if use_val_data and iteration % print_every == 0:
            val_batch = get_batch_val()
            loss_val = []
            for one_batch in val_batch:
                v_batch, type_count, lattice_args = one_batch
                static_args = nn.FrozenDict({'type_count': tuple(type_count),
                                             'lattice': lattice_args})
                loss_val.append(val_step(v_batch, variables, static_args))

        batch, type_count, lattice_args = get_batch_train()
        static_args = nn.FrozenDict({'type_count': tuple(type_count),
                                     'lattice': lattice_args})
        variables, opt_state, state = train_step(batch,
                                                 variables,
                                                 opt_state,
                                                 state,
                                                 static_args)
        
        # training step part 2 in hybrid observable training
        if hybrid and iteration % obs_step_every == 0:
            # observable train step
            for i in range(len(obs_train_data_path)):
                batch, type_count, lattice_args = get_batch_train_obs(obs_position=i)
                static_args = nn.FrozenDict({'type_count': tuple(type_count),
                                            'lattice': lattice_args})
                variables, opt_state, state_obs = train_step_obs(batch,
                                                        variables,
                                                        opt_state,
                                                        state_obs,
                                                        static_args,
                                                        obs_position=i) 

        if iteration % print_every == 0:
            print_step()
            tic = time.time()

    # compress, save, and finish
    if compress:
        model, variables = compress_model(model,
                                                variables,
                                                compress_Ngrids,
                                                compress_r_min)
    save_model(save_path, model, variables)
    print(f'# Training finished in {datetime.timedelta(seconds=int(time.time() - TIC))}.')

def test(
    model_path: str,
    data_path: str,
    batch_size: int = 1,
):
    '''
        Testing a trained model on a **single** dataset.
        Input arguments:
            model_path: path to the trained model.
            data_path: path to the data for evaluation.
            batch_size: Increase for potentially faster evaluation, but requires more memory.
    '''
    if type(data_path) == list:
        raise ValueError('Data_path should be a single string path for now.')

    if jax.device_count() > 1:
        print('# Note: Currently only one device will be used for evaluation.')
    
    model, variables = load_model(model_path, replicate=False)
    if model.params['type'] == 'energy' or model.params['type'] == 'dplr':
        labels = ['coord', 'box', 'force', 'energy']
        atomic_sel = None
    elif 'atomic' in model.params['type']:
        labels = ['coord', 'box', model.params['atomic_data_prefix']]
        atomic_sel = model.params['nsel']
    else:
        raise ValueError('Model type should be "energy", "atomic", "atomic_t2", or "dplr".')
    test_data = DPDataset([data_path],
                          labels,
                          {'atomic_sel':atomic_sel})
    test_data.compute_lattice_candidate(model.params['rcut'])
    if 'dplr' in model.params['type']:
        subsets = test_data.get_flattened_data()
        for subset in subsets:
            process_long_range_subset(subset,
                                      model.params['dplr_q_atoms'],
                                      model.params['dplr_q_wc'],
                                      model.params['dplr_beta'],
                                      model.params['dplr_resolution'],
                                      *model.params['dplr_wannier_model_and_variables'])
    test_data.pointer = 0
    remaining = test_data.nframes
    if model.params['type'] == 'energy' or model.params['type'] == 'dplr':
        evaluate_fn = model.energy_and_force
        predictions = {'energy': [], 'force': []}
        ground_truth = {'energy': [], 'force': []}
    else:
        evaluate_fn = lambda variables, coord, box, static_args: model.apply(variables, coord, box, static_args)
        predictions = {model.params['atomic_data_prefix']: []}
        ground_truth = {model.params['atomic_data_prefix']: []}
    evaluate_fn = jax.jit(jax.vmap(evaluate_fn,
                                   in_axes=(None,0,0,None)),
                                   static_argnames=('static_args',))
    while remaining > 0:
        bs = min(batch_size, remaining)
        batch, type_count, lattice_args = test_data.get_batch(bs)
        remaining -= bs
        static_args = nn.FrozenDict({'type_count': type_count, 'lattice': lattice_args})
        pred = evaluate_fn(variables, batch['coord'], batch['box'], static_args)
        if model.params['type'] == 'energy' or model.params['type'] == 'dplr':
            predictions['energy'].append(pred[0])
            predictions['force'].append(pred[1])
            ground_truth['energy'].append(batch['energy'])
            ground_truth['force'].append(batch['force'])
        else:
            predictions[model.params['atomic_data_prefix']].append(pred[0])
            ground_truth[model.params['atomic_data_prefix']].append(batch['atomic'])
    rmse = {}
    mae = {}
    l1_mixed = {}
    for key in predictions.keys():
        predictions[key] = np.concatenate(predictions[key], axis=0)
        ground_truth[key] = np.concatenate(ground_truth[key], axis=0)
        rmse[key] = (((predictions[key] - ground_truth[key]) ** 2).mean() ** 0.5).item()
        mae[key] = np.abs(predictions[key] - ground_truth[key]).mean().item()
    # reorder force back; will delete in future when reordering is moved into dpmodel.py
    if model.params['type'] == 'energy' or model.params['type'] == 'dplr':
        ground_truth['force'] = ground_truth['force'][:, test_data.type.argsort(kind='stable').argsort(kind='stable')]
        predictions['force'] = predictions['force'][:, test_data.type.argsort(kind='stable').argsort(kind='stable')]
        natoms = predictions['force'].shape[1]
        rmse['energy'] /= natoms
        mae['energy'] /= natoms
        l1_mixed['energy'] = (np.abs(predictions['energy'] - ground_truth['energy']).mean() / natoms).item()
        l1_mixed['force'] = (((predictions['force'] - ground_truth['force'])**2).mean(-1)**0.5).mean().item()
    else:
        key = model.params['atomic_data_prefix']
        l1_mixed[key] = (((predictions[key] - ground_truth[key])**2).mean(-1)**0.5).mean().item()
    return rmse, mae, l1_mixed, predictions, ground_truth
        
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
            type_idx: atomic type indices of shape (Natoms,)
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
            label_dim = 9 if model.params['type'] == 'atomic_t2' else 3
            np.save(atomic_path, np.zeros((coord.shape[0], label_count * label_dim)))
        elif model.params['type'] == 'energy':
            energy_path = os.path.join(set_dir, "energy.npy")
            force_path = os.path.join(set_dir, "force.npy")
            np.save(energy_path, np.zeros(coord.shape[0]))
            np.save(force_path, np.zeros((coord.reshape(coord.shape[0], -1)).shape))
        _, _, _, predictions, _ = test(model_path, temp_dir, batch_size)

    return predictions
    
def process_long_range_subset(subset, dplr_q_atoms, dplr_q_wc, dplr_beta, dplr_resolution, wc_model, wc_variables):
    '''
        subtracting long range energy and force, keeping short range part only, for dplr models.
    '''
    data, type_count, lattice_args = subset.values()
    if not lattice_args['ortho']:
        raise ValueError('For "dplr" currently only orthorhombic boxes are supported.')
    sel_type_count = tuple(np.array(type_count)[wc_model.params['nsel']])
    qatoms = np.repeat(dplr_q_atoms, type_count)
    qwc = np.repeat(dplr_q_wc, sel_type_count)
    static_args = nn.FrozenDict({'type_count': type_count, 'lattice': lattice_args})

    def lr_energy(coord, box, Ngrid):
        wc = wc_model.wc_predict(wc_variables, coord, box, static_args)
        p3mlr_fn = get_p3mlr_fn(jnp.diag(box), dplr_beta, Ngrid)
        return p3mlr_fn(jnp.concatenate([coord, wc]), jnp.concatenate([qatoms, qwc]))
    
    @partial(jax.jit, static_argnums=(2,))
    def lr_energy_and_force(coord, box, Ngrid):
        e, negf = jax.value_and_grad(lr_energy)(coord, box, Ngrid)
        return e, -negf

    for i in range(len(data['coord'])):
        Ngrid = get_p3mlr_grid_size(np.diag(data['box'][i]), dplr_beta, resolution=dplr_resolution)
        e_lr, f_lr = lr_energy_and_force(data['coord'][i], data['box'][i], Ngrid)
        data['energy'][i] -= e_lr
        data['force'][i] -= f_lr
