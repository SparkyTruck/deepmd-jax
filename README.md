# DeepMD-jax

Welcome to **DeepMD-jax v0.2**!

## Supported Features
DeepMD-jax supports:
- **Deep Potential (DP)**: Fast energy and force predictions.
- **Deep Wannier (DW)**: Predicting Wannier centers associated to atoms.
- **DP Long Range (DPLR)**: Incorporate explicit long-range Coulomb interactions.

Also, you can try the **DP-MP** architecture for enhanced accuracy.

Currently allows **NVE/NVT/NPT simulations** on **multiple GPUs** based on a backend of [jax-md](https://github.com/jax-md/jax-md).

## Installation

Requires **CUDA 12** for GPU support (latest CUDA subversion preferred, but not mandatory). We recommend using a conda environment:

```
conda create -n deepmd-jax python=3.12
conda activate deepmd-jax
```

Clone the repository and install it in your working directory:

```
git clone https://github.com/SparkyTruck/deepmd-jax.git --branch staging
cd deepmd-jax
pip install -e .
```
When there is an update, simply do a `git pull` in the `deepmd-jax` directory.

## Quick Start

### Preparing Your Dataset
To train a model, prepare your dataset in the same [DeepMD-kit format](https://docs.deepmodeling.com/projects/deepmd/en/r2/data/system.html). Note: Currently only supports periodic systems.


### Training a Model
Once your dataset is ready, train a model like this:

```python
from deepmd_jax.train import train

train(
      model_type='energy',             # Model type: 'energy', 'atomic', 'dplr'
      rcut=6.0,                        # Cutoff radius
      save_path='trained_model.pkl',   # Path to save the trained model
      train_data_path='dataset/path/', # Path (or a list of paths) to the training dataset
      step=1000000,                    # Number of training steps
)
```

The default hyperparameters should be an okay baseline, but you can adjust additional arguments in `train()`, such as `mp=True` to use DP-MP, and `lr`, `batch_size`, `embed_widths`, etc.

### Evaluating a Model

You can evaluate the model on a dataset with `test()`:
```python
from deepmd_jax.train import test
rmse, predictions, ground_truth = test(model_path, data_path)
```

Or use `evaluate()` on a set of configurations, where no ground truth is needed:
```python
from deepmd_jax.train import evaluate
predictions = evaluate(model_path, coord, box, type_idx)
```

### Running a Simulation

To run a simulation, prepare the following numpy arrays:

1. `initial_position`: shape `(n, 3)` where `n` is the number of atoms.
2. `box`: shape `(1,)`, `(3,)` or `(3, 3)`.
3. `type_idx`: shape `(n,)`, indicates the type of each atom (similar to `type.raw` in the training dataset).

Then, an example of running a simulation is as follows:

```python
from deepmd_jax.md import Simulate

sim = Simulate(
    model_path='trained_model.pkl',
    box=box,                           # Angstroms
    type_idx=type_idx,                 # index-element map (i.e. 0 - Oxygen, 1 - Hydrogen) must match the dataset used to train the model
    mass=[15.9994, 1.0078],            # Oxygen, Hydrogen
    routine='NVT',                     # 'NVE', 'NVT', 'NPT'
    dt=0.5,                            # femtoseconds
    initial_position=initial_position, # Angstroms
    temperature=300,                   # Kelvin
)

trajectory = sim.run(100000)           # Run for 100,000 steps
```

Again, check the `Simulate` class for additional arguments and methods.

### Precision Settings

By default, single precision `float32` is used for both training and simulation, which I find to be generally sufficient. However, if you need double precision, enable it at the **beginning** of your script with:

```python
import jax
jax.config.update('jax_enable_x64', True)
```

### Units

The default units are Angstrom, eV, femtosecond, and their derived units. The only exceptions are the parameters `temperature` (Kelvin), `pressure` (bar), and `mass` (Dalton) when initializing `Simulate()`.

## Roadmap

A tentative to-do list (in no particular order):
- [ ] Optimize training and simulation when neighbor lists are not used.
- [ ] Misc: data, dpmodel, utils code cleanup; Glob data path, optimize compute lattice; move reorder inside dpmodel; train starting from a trained model; training seed control; print log redirect.
- [ ] Multi-host large scale simulation support.
- [ ] Misc simulation features: Custom energy functions, time-dependent potentials, temperature and pressure control, fix atoms, more thermostats.
- [ ] DWIR support.
- [ ] Further tune NN architecture and training hyperparameters (v0.2.1).

Future considerations:
- [ ] Enhanced sampling. (v0.3)

This project is in active development, and if you encounter any issues, please feel free to contact me or open an issue on the GitHub page. Have fun! 🚀

