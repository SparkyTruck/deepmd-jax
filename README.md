# DeepMD-jax

Welcome to **DeepMD-jax v0.2**!

## Supported Features
DeepMD-jax supports:
- **Deep Potential (DP)**: Fast energy and force predictions.
- **Deep Wannier (DW)**: Predicting Wannier centers associated with atoms.
- **DP Long Range (DPLR)**: Explicit long-range Coulomb interactions.
- **Hybrid ab initio and empirical DP**: Empirical-observable training ([link to article](https://arxiv.org/abs/2511.14352)).
- **Classical MD and PIMD**: Built-in `NVE`, `NVT`, `NVT_langevin`, and `NPT` simulation, including NVT/NPT path-integral MD.

You can also try the [**DP-MP**](https://pubs.rsc.org/en/content/articlehtml/2024/cp/d4cp01483a) architecture for enhanced accuracy. Simulations can run on **multiple GPUs** using [jax-md](https://github.com/jax-md/jax-md)-based routines.

## Installation
```
git clone https://github.com/SparkyTruck/deepmd-jax.git
cd deepmd-jax
pip install -e .
```

## Hardware Requirements

It is recommended to have one GPU for training and one or more GPUs for simulation. The RTX 4090/5090 is recommended based on cost-effectiveness for most fp32 jobs.

## Quick Start

### Step 1: Prepare Your Dataset
Training data can be either [DeepMD-kit format](https://docs.deepmodeling.com/projects/deepmd/en/r2/data/system.html) directories or ASE-readable extended XYZ files (`.xyz`/`.extxyz`). For energy models, extxyz frames should contain lattice/cell, species, positions, energy, and forces. Do not mix DeepMD directories and extxyz files in the same `train_data_path` list. Currently only periodic systems are supported.

### Step 2: Train a Deep Potential Force Field

```python
from deepmd_jax.train import train

train(
      model_type='energy',                   # Model type
      rcut=6.0,                              # Cutoff radius
      save_path='model.pkl',                 # Path to save the trained model
      train_data_path='/energy/force/data/', # Path (or a list of paths) to the training dataset
      step=1000000,                          # Number of training steps
)
```

The default values for the other arguments in [`train()`](https://github.com/SparkyTruck/deepmd-jax/blob/main/deepmd_jax/train.py) like learning rate, batch size, model width, etc. are usually a solid baseline. The one parameter you may want to change is `mp=True` to enable DP-MP for better accuracy.

### Step 3: Perform a Simulation

Prepare numpy arrays for `initial_position` `(n, 3)`, `box` `()`, `(1,)`, `(3,)`, or `(3, 3)`, and `type_idx` `(n,)`. For DeepMD-format models, `type_idx` uses type indices matching `type.raw`; for extxyz-trained models, it can use atomic numbers. `mass` is per model type, following stored `chemical_types` order for extxyz models.

```python
from deepmd_jax.md import Simulation

sim = Simulation(
    model_path='model.pkl',
    box=box,
    type_idx=type_idx,
    mass=[15.9994, 1.0078],
    routine='NVT',                     # 'NVE', 'NVT', 'NVT_langevin', or 'NPT'
    dt=0.5,                            # femtoseconds
    initial_position=initial_position, # Angstroms
    temperature=300,                   # Kelvin; required for NVT/NPT/PIMD
)

trajectory = sim.run(10000)
print(trajectory['position'].shape)    # (10001, n, 3), includes the initial frame
trajectory = sim.run(10000)
print(trajectory['position'].shape)    # (10000, n, 3), continuation omits the old initial frame
```

Routine choices:
- `NVE`: velocity-Verlet, no thermostat.
- `NVT`: Nose-Hoover thermostat.
- `NVT_langevin`: BAOAB Langevin thermostat, with friction `1/tau_t`.
- `NPT`: Nose-Hoover thermostat/barostat; set `pressure` in bar. `couple_axes` controls isotropic, semi-isotropic, or anisotropic orthorhombic box fluctuations.

For PIMD, set `n_bead > 1` with `routine='NVT'` or `routine='NPT'`:

```python
sim = Simulation(
    model_path='model.pkl',
    box=box,
    type_idx=type_idx,
    mass=[15.9994, 1.0078],
    routine='NVT',
    dt=0.25,
    initial_position=initial_position, # (n, 3) is replicated to beads
    temperature=300,
    n_bead=16,                         # number of ring-polymer beads
)
```

PIMD `NVT` uses a PILE-L Langevin thermostat; PIMD `NPT` uses PILE-L plus a Langevin barostat. See [`md.py`](https://github.com/SparkyTruck/deepmd-jax/blob/main/deepmd_jax/md.py) for extra controls such as `tau_t`, `tau_p`, `neighbor_skin`, `fixed_indices`, and `couple_axes`.

## More Features and Usages

### Testing a Model

Use `test()` to evaluate a model on a dataset:
```python
from deepmd_jax.train import test
root_mean_sq_err, mean_abs_error, l1_mixed_error, predictions, ground_truth = test(model_path, data_path)
```

### Evaluating a Model

Use `evaluate()` on a batch of configurations where no ground truth is needed:
```python
from deepmd_jax.train import evaluate
predictions = evaluate(model_path, coords, boxes, type_idx)
```

### Atom-wise vector prediction (Deep Wannier/Polarizability)
```python
train(
      model_type='atomic',                   # Model type, 'atomic' for 3-vector (Wannier centroid), 'atomic-t2' for 9-vector (polarizability)
      rcut=6.0,                              # Cutoff radius
      atomic_sel=[0],                        # indicating Wannier centers are associated to atoms of type 0
      save_path='wannier.pkl',               # Path to save the trained model
      train_data_path='/wannier/data/',      # Path (or a list of paths) to the training dataset
      step=100000,                           # Number of training steps
)
```
This model predicts an O(3)-equivariant 3-vector or 9-vector (as a flattened symmetric 3x3 matrix) for each atom within the selected type (`atomic_sel`). For `model_type='atomic'`, the argument `atomic_data_prefix` in `train()` is set to `'atomic_dipole'` by default, meaning `"atomic_dipole.npy"` is the expected label filename in your dataset. For `model_type='atomic_t2'`, it defaults to `'atomic_polarizability'`. If you need to specify a different prefix, it needs to be `'atomic_*'`.

### Training a DPLR Model
```python
# train a Wannier model first and then train DPLR
train(
      model_type='dplr',                     # Model type
      rcut = 6.0,                            # Cutoff radius
      save_path = 'dplr_model.pkl',          # Path to save the trained model
      dplr_wannier_model_path='wannier.pkl', # Path to the trained Wannier model
      train_data_path='/energy/force/data/', # Path (or a list of paths) to the training dataset
      step=1000000,                          # Number of training steps
      dplr_q_atoms=[6, 1],                   # Charge of atomic cores of each type (eg. oxygen and hydrogen)
      dplr_q_wc = [-8],                      # Charge of Wannier centers associated to each atom in atomic_sel type
)
```

### Training a Hybrid Ab Initio and Empirical Model
```python
train(
      model_type='energy',                   # Model type
      hybrid=True,                           # Set hybrid to true
      rcut = 6.0,                            # Cutoff radius
      save_path = 'model.pkl',               # Path to save the trained model
      train_data_path='/energy/force/data/', # Path (or a list of paths) to the training dataset
      step=1000000,                          # Number of training steps
      obs_train_data_path = '/hybrid/data',  # Path (or a list of paths) with observable dataset
      obs_temperature = 320,                 # Temperature in Kelvin corresponding to observable dataset
      obs_target = 1.0,                      # Target (empirical) value of the observable
      obs_batch_size = 100,                  # Batch size for observable loss. Usually >> 1 to allow reweighting
      obs_s_pref = 0.02,                     # Starting value of prefactor in the observable loss term
      obs_l_pref = 100,                      # Last value of prefactor in the observable loss term
)
```

### Running with ASE

An ASE calculator is provided and can be used to run energy minimizations or molecular dynamics. This is a minimal example with the same `initial_position`, `type_idx`, and `box` of shape `(3, 3)` or `(3,)`:
```python
from deepmd_jax.md import DPJaxCalculator
from ase import Atoms
from ase.md.langevin import Langevin
from ase import units

calc = DPJaxCalculator(model_path="./model.pkl", type_idx=type_idx)

type_map = {0: "O", 1: "H"}
symbols = [type_map[i] for i in type_idx]
atoms = Atoms(
    symbols=symbols,
    positions=initial_position,
    cell=box,
    pbc=True
)

atoms.set_calculator(calc)

dyn = Langevin(atoms, timestep=0.5 * units.fs, temperature_K=300, friction=0.01)
dyn.run(10)  # 10 MD steps
```

The ASE calculator may run slower than the built-in `Simulation`, but you can take advantage of the rich ASE features, such as minimization routines, different thermostats, non-isotropic NPT simulations, etc.

### Precision Settings

By default, single precision `float32` is used for both training and simulation, which I find to be generally sufficient. However, if you need double precision, enable it at the **beginning** of your script with:

```python
import jax
jax.config.update('jax_enable_x64', True)
```

### Units

The default units are Angstrom, eV, femtosecond, and their derived units. The only exceptions are the parameters `temperature` (Kelvin), `pressure` (bar), and `mass` (Dalton) when initializing `Simulation()`.

### Printing Trajectories on the Fly

Pass `dump_prefix` to stream XYZ files instead of keeping the trajectory in memory. The generated files are `{prefix}_position.xyz`, `{prefix}_velocity.xyz`, and, for PIMD, `{prefix}_centroid.xyz`. Each XYZ frame includes `step=<self.step>` in the metadata line. Existing files are overwritten by default; use `dump_mode='append'` to continue an existing trajectory.

```python
# Writes traj_position.xyz and traj_velocity.xyz every 10 steps.
sim.run(100000, dump_prefix="traj", dump_interval=10)

# Dump a subset. Allowed entries are 'position', 'velocity', and 'centroid'.
sim.run(100000, dump_prefix="traj_pos", dump_content=["position"], dump_interval=10)

# Continue appending frames to existing files.
sim.run(100000, dump_prefix="traj", dump_interval=10, dump_mode="append")
```

For XYZ output, pass `type_symbols=[...]` to `Simulation` unless the model was trained from extxyz and stores `chemical_types`.

## Roadmap

To-do list:
- [ ] Model deviation API.
- [ ] Non-orthorhombic neighbor list and broader NPT cell support.
- [ ] Enhanced sampling and additional thermostats/barostats.
- [ ] Misc: data/model utility cleanup, pair correlation function, training from a saved model, and training seed control.

This project is in active development, and if you encounter any issues, please feel free to contact me or open an issue on the GitHub page. You are also welcome to make custom modifications and pull requests. Have fun! 🚀

## Troubleshooting

If jax doesn't see a GPU when there is one, it could be due to the environment variable `LD_LIBRARY_PATH` not pointing to the right CUDA libraries. The simplest fix is running `unset LD_LIBRARY_PATH` in bash before launching python.
For jupyter notebooks, you can install a kernel with
```bash
python -m ipykernel install --user --name deepmd-jax-cuda12 --display-name "Python (deepmd-jax-cuda12)" --env LD_LIBRARY_PATH ""
```
In certain HPC environments, an alternative solution is to `module load` a latest CUDA 12 version.
