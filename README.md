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

## Quick Start

### Preparing Your Dataset
To train a model, prepare your dataset in the same [DeepMD-kit format](https://docs.deepmodeling.com/projects/deepmd/en/r2/data/system.html). Note: Currently only supports periodic systems.


### Training a Model
Once your dataset is ready, train a model like this:

```python
from deepmd_jax.train import train

train(
      model_type='energy',             # Model type: 'energy' for energy models, 'atomic' for Wannier models
      rcut=6.0,                        # Cutoff radius
      save_path='trained_model.pkl',   # Path to save the trained model
      train_data_path='dataset/path/', # Path to the training dataset, or you can use a list of paths
      step=1000000,                    # Number of training steps
)
```

The default hyperparameters should be an okay baseline, but you can adjust additional arguments in `train()`, such as `mp=True` to use DP-MP, and `lr`, `batch_size`, `embed_widths`, etc.

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
    type_idx=type_idx,                 # here the map (i.e. 0 for Oxygen, 1 for Hydrogen) must match the dataset used to train the model
    mass=[15.9994, 1.0078],            # Oxygen, Hydrogen
    routine='NVT',                     # 'NVE', 'NVT', 'NPT'
    dt=0.5,                            # femtoseconds
    initial_position=initial_position, # Angstroms
    temperature=300,                   # Kelvin
)

trajectory = sim.run(100000)           # Run for 100,000 steps
```

Again, check the `Simulate` class for possible additional arguments.

### Precision Settings

By default, single precision (`float32`) is used for both training and simulation, which I find to be generally sufficient. However, if you need double precision, enable it at the **beginning** of your script with:

```python
import jax
jax.config.update('jax_enable_x64', True)
```

### 

We're in active development, and if you encounter any issues, please feel free to contact me or open an issue on the GitHub page. Have fun!