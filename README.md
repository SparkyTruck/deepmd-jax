# Deepmd-JAX

Welcome to **Deepmd-JAX v0.2**!

## Supported Features
Deepmd-JAX supports:
- **Deep Potential (DP)**: Fast energy and force predictions.
- **Deep Wannier (DW)**: Predicting Wannier centers associated to atoms.
- **DP Long Range (DPLR)**: Incorporate explicit long-range Coulomb interactions.

Also, you can try the **DP-MP** architecture for enhanced accuracy.

Currently allows **NVE/NVT/NPT simulations** on multiple GPUs with a jax-md backend. offering a simple, lightning-fast training and simulation workflow.

## Installation

Requires **CUDA 12** for GPU support (latest CUDA subversion preferred, but not mandatory). We recommend using a conda environment:

```
conda create -n deepmd-jax python=3.12`  
conda activate deepmd-jax
```

Clone the repository and install it in your working directory:

```
git clone https://github.com/SparkyTruck/deepmd-jax.git
pip install -e ./deepmd-jax
```

## Quick Start

### Preparing Your Dataset
To train a model, prepare a dataset in the DeepMD-kit format as described in the [DeepMD-kit documentation](https://docs.deepmodeling.com/projects/deepmd/en/r2/data/system.html). **Note**: Currently only supports periodic systems.


### Training a Model
Once your dataset is ready, do something like

```python
from deepmd_jax.train import train

train(
      model_type='energy',       # Model type: 'energy' for energy models
      rcut=6.0,                  # Cutoff radius
      save_path='trained_model.pkl',  # Path to save the trained model
      train_data_path='dataset/path/', # Path to the dataset
      step=500000                # Number of training steps
)
```

The default hyperparameters should be an okay baseline, but you can customize additional arguments in `train()`, such as `atomic` (for Wannier models), `mp=True` (for DP-MP), and adjust `lr`, `batch_size`, `embed_widths`, etc.

### Running a Simulation

To run a simulation, prepare the following NumPy arrays:

1. **Initial Position** (`Å`): A 2D array of shape `(n, 3)` where `n` is the number of atoms.
2. **Box** (`Å`): Either a 1D array of shape `(3,)` or a 2D array of shape `(3, 3)` to specify the simulation box dimensions.
3. **Type Index**: A 1D array of shape `(n,)` that indicates the type of each atom (similar to `type.raw` in the training dataset).

Then, use the following code to run the simulation:

```python
from deepmd_jax.md import Simulate

sim = Simulate(
    model_path='trained_model.pkl',
    box=box,
    type_idx=type_idx,
    mass=[15.9994, 1.0078],  # Example: O and H
    routine='NVT',           # Options: 'NVE', 'NVT', 'NPT'
    dt=0.5,                  # Timestep in femtoseconds
    initial_position=initial_position,
    temperature=350,         # Temperature in Kelvin
)

trajectory = sim.run(100000)  # Run for 100,000 steps
```

### Precision Settings

By default, Deepmd-JAX uses single precision (`float32`) for both training and simulation, which is generally sufficient for most cases. However, if you need double precision, enable it at the *beginning* of your script with:

```python
import jax
jax.config.update('jax_enable_x64', True)
```

If you encounter any issues, please feel free to contact me or open an issue on the GitHub page. Have fun!