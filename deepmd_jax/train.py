import jax
import optax
import numpy as np
import jax.numpy as jnp
import time, datetime
import flax.linen as nn
from functools import partial
from deepmd_jax import data, utils
from deepmd_jax.dpmodel import DPModel
from typing import Union, List

def train(
    model_type: str,
    rcut: float,
    save_path: str,
    train_data_path: Union[str, List[str]],
    val_data_path: Union[str, List[str]] = None,
    step: int = 1000000,
    mp: bool = False,
    atomic_sel: List[int] = None,
    embed_widths: List[int] = [24,48,96],
    embed_mp_widths: List[int] = [64,64,64],
    fit_widths: List[int] = [128,128,128],
    axis_neurons: int=12,
    lr: float = None,
    batch_size: int = None,
    val_batch_size: int = None,
    compress: bool = True,
    print_every: int = 1000,
    atomic_data_prefix: str = 'atomic_dipole',
    s_pref_e: float = 0.02,
    l_pref_e: float = 1,
    s_pref_f: float = 1000,
    l_pref_f: float = 1,
    lr_limit: float = 5e-7,
    beta2: float = 0.99,
    decay_steps: int = 5000,
    getstat_bs: int = 64,
    label_bs: int = 256,
    tensor_2nd: bool = True,
    print_loss_smoothing: int = 20,
    compress_Ngrids: int = 1024,
    compress_r_min: float = 0.6,
    seed: int = None
):
    '''
        Entry point for training deepmd-jax models.
        Input arguments:
            model_type: either 'energy' (for standard force field) or 'atomic' (atomic property, e,g. Deep Wannier).
            rcut: cutoff radius (Angstrom) for the model.
            save_path: path to save the trained model.
            train_data: path to training data (str) or list of paths to training data (List[str]).
            val_data: path to validation data (str) or list of paths to validation data (List[str]).
            step: number of training steps.
            mp: whether to use message passing model for more accuracy at a higher cost.
            atomic_sel: Selects the atom types for prediction. Only used when model_type == 'atomic'.
            embed_widths: Widths of the embedding neural network.
            embed_mp_widths: Widths of the embedding neural network in message passing. Only used when mp == True.
            fit_widths: Widths of the fitting neural network.
            axis_neurons: Number of axis neurons to project the atomic features before the fitting network. Recommended range: 8-16.
            lr: learning rate at start. If None, default values (0.002 for 'energy' and 0.01 for 'atomic') is used.
            batch_size: training batch size in number of frames. If None, will be automatically determined by label_bs.
            val_batch_size: validation batch size in number of frames. If None, will be automatically determined by 4*label_bs.
            compress: whether to compress the model after training for faster inference.
            print_every: interval for printing loss and validation.
            atomic_data_prefix: prefix for .npy label files when model_type == 'atomic'.
            s_pref_e: starting prefactor for energy loss.
            l_pref_e: limit prefactor for energy loss.
            s_pref_f: starting prefactor for force loss.
            l_pref_f: limit prefactor for force loss.
            lr_limit: learning rate at end of training.
            beta2: adam optimizer parameter.
            decay_steps: learning rate exponentially decays every decay_steps.
            getstat_bs: batch size for computing model statistics at initialization.
            label_bs: training batch size in number of atoms.
            tensor_2nd: whether to use 2nd order tensor descriptor for more accuracy.
            print_loss_smoothing: smoothing factor for loss printing.
            compress_Ngrids: Number of intervals used in compression.
            compress_r_min: A safe lower bound for interatomic distance in the compressed model.
    '''
    
    TIC = time.time()
    if jax.device_count() > 1:
        print('# Note: Currently only one device will be used for training.')

    # load dataset
    if model_type == 'energy':
        labels = ['coord', 'box', 'force', 'energy']
    elif model_type == 'atomic':
        labels = ['coord', 'box', atomic_data_prefix]
        assert atomic_sel is not None, ' Must provided atomic_sel for model_type "atomic".'
    else:
        raise ValueError('Model_type should be either "energy" or "atomic".')
    if type(train_data_path) == str:
        train_data_path = [train_data_path]
    train_data = data.DPDataset(train_data_path,
                                labels,
                                {'atomic_sel':atomic_sel})
    train_data.compute_lattice_candidate(rcut)
    use_val_data = val_data_path is not None
    if use_val_data:
        if type(val_data_path) == str:
            val_data_path = [val_data_path]
        val_data = data.DPDataset(val_data_path,
                                  labels,
                                  {'atomic_sel':atomic_sel})
        val_data.compute_lattice_candidate(rcut)
    else:
        val_data = None

    # construct model
    params = {
        'embed_widths': embed_widths[:-1] if mp else embed_widths,
        'embedMP_widths': embed_widths[-1:] + embed_mp_widths if mp else None,
        'fit_widths': fit_widths,
        'axis': axis_neurons,
        'Ebias': train_data.fit_energy() if model_type == 'energy' else None,
        'rcut': rcut,
        'use_2nd': tensor_2nd,
        'use_mp': mp,
        'atomic': model_type == 'atomic',
        'nsel': atomic_sel if model_type == 'atomic' else None,
        'out_norm': 1. if model_type == 'energy' else train_data.get_atomic_label_scale(),
        **train_data.get_stats(rcut, getstat_bs),
    }
    model = DPModel(params)
    print('# Model params:', model.params)

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
        lr = 0.002 if model_type == 'energy' else 0.01
    if step < decay_steps * 10:
        decay_steps = step // 10
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
    loss, loss_and_grad = model.get_loss_fn()
    if model_type == 'energy':
        state = {'loss_avg': 0., 'le_avg': 0., 'lf_avg': 0., 'iteration': 0}
    else:
        state = {'loss_avg': 0., 'iteration': 0}
    
    @partial(jax.jit, static_argnames=('static_args',))
    def train_step(batch, variables, opt_state, state, static_args):
        r = lr_scheduler(state['iteration']) / lr
        if model_type == 'energy':
            pref = {'e': s_pref_e*r + l_pref_e*(1-r),
                    'f': s_pref_f*r + l_pref_f*(1-r)}
            (loss_total, (loss_e, loss_f)), grads = loss_and_grad(variables,
                                                                  batch,
                                                                  pref,
                                                                  static_args)
            for key, value in zip(['loss_avg', 'le_avg', 'lf_avg'],
                                  [loss_total, loss_e, loss_f]):
                state[key] = state[key] * (1-1/print_loss_smoothing) + value
        else:
            loss_total, grads = loss_and_grad(variables,
                                              batch,
                                              static_args)
            state['loss_avg'] = state['loss_avg'] * (1-1/print_loss_smoothing) + loss_total
        updates, opt_state = optimizer.update(grads, opt_state, variables)
        variables = optax.apply_updates(variables, updates)
        state['iteration'] += 1
        return variables, opt_state, state
    
    # define validation step
    @partial(jax.jit, static_argnames=('static_args',))
    def val_step(batch, variables, static_args):
        if model_type == 'energy':
            pref = {'e': 1, 'f': 1}
            _, (loss_e, loss_f) = loss(variables,
                                       batch,
                                       pref,
                                       static_args)
            return loss_e, loss_f
        else:
            loss_total = loss(variables,
                              batch,
                              static_args)
            return loss_total
        
    # define print step
    def print_step():
        beta_smoothing = print_loss_smoothing * (1 - (1/print_loss_smoothing) ** (iteration+1))
        line = f'Iter {iteration:7d}'
        line += f' L {(state["loss_avg"] / beta_smoothing) ** 0.5:7.5f}'
        if model_type == 'energy':
            line += f' LE {(state["le_avg"] / beta_smoothing) ** 0.5:7.5f}'
            line += f' LF {(state["lf_avg"] / beta_smoothing) ** 0.5:7.5f}'
        if use_val_data:
            if val_batch_size is None:
                val_batch, type_count, lattice_args = val_data.get_batch(4 * label_bs, 'label')
            else:
                val_batch, type_count, lattice_args = val_data.get_batch(val_batch_size)
            static_args = nn.FrozenDict({'type_count': tuple(type_count), 'lattice': lattice_args})
            loss_val = val_step(val_batch, variables, static_args)
            if model_type == 'energy':
                line += f' LEval {loss_val[0] ** 0.5:7.5f}'
                line += f' LFval {loss_val[1] ** 0.5:7.5f}'
            else:
                line += f' Lval {loss_val ** 0.5:7.5f}'
        line += f' Time {time.time() - tic:.2f}s'
        print(line)
    
    # training loop
    tic = time.time()
    for iteration in range(step+1):
        if batch_size is None:
            if iteration == 0:
                print(f'# Auto batch size = int({label_bs}/nlabels_per_frame)')
            batch, type_count, lattice_args = train_data.get_batch(label_bs, 'label')
        else:
            if iteration == 0:
                print(f'# Batch size = {batch_size}')
            batch, type_count, lattice_args = train_data.get_batch(batch_size)
        static_args = nn.FrozenDict({'type_count': tuple(type_count),
                                     'lattice': lattice_args})
        variables, opt_state, state = train_step(batch,
                                                 variables,
                                                 opt_state,
                                                 state,
                                                 static_args)
        if iteration % print_every == 0:
            print_step()
            tic = time.time()

    # compress model
    if compress:
        model, variables = utils.compress_model(model,
                                                variables,
                                                compress_Ngrids,
                                                compress_r_min)
        
    # save model and finish
    utils.save_model(save_path, model, variables)
    print(f'# Training finished in {datetime.timedelta(seconds=int(time.time() - TIC))}.')



        

    



    
    
    
    
        



          
