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
```
git clone https://github.com/SparkyTruck/deepmd-jax.git
cd deepmd-jax
pip install -e .
```


## Quick Start

### Preparing Your Dataset
To train a model, prepare your dataset in the same [DeepMD-kit format](https://docs.deepmodeling.com/projects/deepmd/en/r2/data/system.html). Note: Currently only supports periodic systems.


### Training a Model

```python
from deepmd_jax.train import train
```

#### Training an energy-force model
```python
train(
      model_type='energy',                   # Model type
      rcut=6.0,                              # Cutoff radius
      save_path='model.pkl',                 # Path to save the trained model
      train_data_path='/energy/force/data/', # Path (or a list of paths) to the training dataset
      step=1000000,                          # Number of training steps
)
```

#### Training a Wannier model
```python
train(
      model_type='atomic',                   # Model type
      rcut=6.0,                              # Cutoff radius
      atomic_sel=[0],                        # indicating Wannier centers are associated to atoms of type 0
      save_path='wannier.pkl',               # Path to save the trained model
      train_data_path='/wannier/data/',      # Path (or a list of paths) to the training dataset
      step=100000,                           # Number of training steps
)
# default data file prefix for Wannier centroids is "atomic_dipole.npy"
```

#### Training a DPLR model:
```python
# train Wannier first and then train DPLR
train(
      model_type='dplr',                     # Model type
      rcut = 6.0,                            # Cutoff radius
      save_path = 'dplr_model.pkl',          # Path to save the trained model
      dplr_wannier_model_path='wannier.pkl', # Path to the trained Wannier model
      train_data_path='/energy/force/data/', # Path (or a list of paths) to the training dataset
      step=1000000,                          # Number of training steps
      dplr_q_atoms=[6, 1],                   # Charge of atoms of each type
      dplr_q_wc = [-8],                      # Charge of Wannier centers of each atomic_sel type
)
```

In `train()`, set `mp=True` to enable DP-MP for better accuracy; the default values for the other arguments in `train()` like learning rate, batch size, model width, etc. are usually a solid baseline.

### Evaluating a Model

You can evaluate the model with `test()` or `evaluate()`:
```python
from deepmd_jax.train import test, evaluate
# use test() on a dataset
rmse, predictions, ground_truth = test(model_path, data_path)
# use evaluate() on a batch of configurations where no ground truth is needed
predictions = evaluate(model_path, coords, boxes, type_idx)
```

### Running a Simulation

To run a simulation, prepare the following numpy arrays:

1. `initial_position`: shape `(n, 3)` where `n` is the number of atoms.
2. `box`: shape `(,)`, `(1,)`, `(3,)` or `(3, 3)`.
3. `type_idx`: shape `(n,)`, indicates the type of each atom (similar to `type.raw` in the training dataset).

Then, an example of running a simulation is as follows:

```python
from deepmd_jax.md import Simulation

sim = Simulation(
    model_path='trained_model.pkl',    # Has to be an 'energy' or 'dplr' model
    box=box,                           # Angstroms
    type_idx=type_idx,                 # here the index-element map (e.g. 0-Oxygen, 1-Hydrogen) must match the dataset used to train the model
    mass=[15.9994, 1.0078],            # Oxygen, Hydrogen
    routine='NVT',                     # 'NVE', 'NVT', 'NPT' (Nosé-Hoover)
    dt=0.5,                            # femtoseconds
    initial_position=initial_position, # Angstroms
    temperature=300,                   # Kelvin
)

trajectory = sim.run(10000)            # Run for 10,000 steps
print(trajectory['position'].shape)    # (100001, n, 3)
# you can split into multiple runs if needed
trajectory = sim.run(10000)            # Continue to run another 10,000 steps
print(trajectory['position'].shape)    # (100000, n, 3), does not include the initial position
```

You can check the `Simulation` class for additional initialization arguments, like print control, thermostat parameters, etc. There are also some methods of the `Simulation` class like `getEnergy`, `getForces`, `getPosition`, `setPosition`, etc.

If you want to print the trajectories on the fly, you can use the `TrajDumpSimulation` instead of `Simulation`:
```python
from deepmd_jax.md import TrajDump, TrajDumpSimulation

sim = TrajDumpSimulation(
    model_path="model.pkl",  # Has to be an 'energy' or 'dplr' model
    box=box,  # Angstroms
    type_idx=type_idx,  # here the index-element map (e.g. 0-Oxygen, 1-Hydrogen) must match the dataset used to train the model
    mass=[15.9994, 1.0078, 195.08],  # Oxygen, Hydrogen
    routine="NVT",  # 'NVE', 'NVT', 'NPT' (Nosé-Hoover)
    dt=0.5,  # femtoseconds
    initial_position=initial_position,  # Angstroms
    temperature=330,  # Kelvin
    report_interval=100,  # Report every 100 steps
)
# print positions and velocities every 10 steps in xyz format
sim.run(
      n_steps,
      [
      TrajDump(atoms, "pos_traj.xyz", 10, append=True),
      TrajDump(atoms, "vel_traj.xyz", 10, vel=True, append=True),
      ],
)
# Run for 100,000 steps
trajectory = sim.run(100000)
```

### Precision Settings

By default, single precision `float32` is used for both training and simulation, which I find to be generally sufficient. However, if you need double precision, enable it at the **beginning** of your script with:

```python
import jax
jax.config.update('jax_enable_x64', True)
```

### Units

The default units are Angstrom, eV, femtosecond, and their derived units. The only exceptions are the parameters `temperature` (Kelvin), `pressure` (bar), and `mass` (Dalton) when initializing `Simulation()`.

## Roadmap

To-do list:
- [ ] Fix atoms/dummy atoms; Optimize multi-gpu sharding.
- [ ] Model deviation API; evaluate DPLR;
- [ ] Misc simulation features: Temperature and pressure control, more thermostats, remove center of mass motion; 
- [ ] Optimize NPT speed and memory usage (could be a jax-md issue)
- [ ] DWIR support (iterative refinement).

Planned features: (v0.3)
- [ ] Enhanced sampling. 
- [ ] Path-Integral MD.
- [ ] Non-orthorhomibic neighbor list; Non-isotropic fluctuation in NPT.
- [ ] Misc: data, dpmodel, utils code cleanup; Glob data path, flatten subset, optimize compute lattice, optimize print output; pair correlation function; move reorder inside dpmodel; train starting from a trained model; training seed control; 

This project is in active development, and if you encounter any issues, please feel free to contact me or open an issue on the GitHub page. You are also welcome to make custom modifications and pull requests. Have fun! 🚀

## Troubleshooting

In certain HPC environments, if jax doesn't see a GPU when there is one, you may need to `module load` a latest CUDA 12 version. To solve this environment problem in jupyter notebooks, you can install a kernel with the right environment variables set. For example:
```bash
python -m ipykernel install --user --name deepmd-jax-cuda12 --display-name "Python (deepmd-jax-cuda12)" --env LD_LIBRARY_PATH ""
```

