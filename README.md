# Deepmd-JAX

Supports Deep Potential, Deep Wannier, and Long Range (DPLR) models. Also, try the Deep Potential Message Passing (DP-MP) model for more accuracy. 

## Installation
We recommend using conda to create a new environment with the latest GPU version of JAX installed.
```bash
conda create --name deepmd-jax python=3.10
conda activate deepmd-jax 
pip install --upgrade "jax[cuda12_pip]"==0.4.26 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Then, download the package to some path, and install it by
```
cd your_path/deepmd-jax
conda activate deepmd-jax 
pip install -e .
```

To run in Jupyter notebooks, create a kernel by
```
conda activate deepmd-jax 
pip install ipykernel
python -m ipykernel install --user --name deepmd-jax
```

## Example scripts
Examples are mostly based on an H2O system.
- `examples/train.py`: Training a model. Note: You should prepare your data in the format of DeepMD-kit.
- `examples/simulate.py`: Simulation in NVT/NVE with trained models.
- `examples/evaluate.py`: Evaluating model predictions.
- `examples/convert_dplr_dataset.py`: Convert dataset to short range part used in DPLR.
- `examples/sub`: Example slurms scripts for running on clusters.

You can modify the paramters inside the the scripts to fit your needs, or create your own script from it.

Note: In the script `examples/simulate.py`, multiple GPU/TPU devices can be used, but it only runs on a single compute node.
