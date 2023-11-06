# Deepmd-JAX

Currently supports Deep Potential and Deep Wannier models.  Also, try the Deep Potential Message Passing (DP-MP) model for more accuracy. The package is still under development and the API can change in the future.

## Setting up environment
```bash
conda create --name deepmd-jax python=3.10
conda activate deepmd-jax 
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax optax jax-md jaxopt matplotlib
```
Please download the package and see the scripts in `examples/` folder. No installation required. 

To run in Jupyter notebooks, create a kernel by
```
conda activate deepmd-jax 
pip install ipykernel
python -m ipykernel install --user --name deepmd-jax
```

## Example scripts
- `examples/train.py`: train a model
- `examples/evaluate.ipynb`: evaluate model predictions (These scripts should execute in seconds on GPUs)
- `examples/simulate.py`: simulate a system in NVT/NVE ensemble with trained models
- `sub` and `sub_simulation`: example slurms scripts for running on clusters (here it is Perlmutter at NERSC).

You can modify the paramters inside the the scripts to fit your needs, or create you're own script from it.

Note: In training, no neighbor list need to be used. In simulations, neighbor lists are only supported when the periodic box is orthorhombic and the model cutoff is less than half of the box length. For simulation with neighbor lists, currently multiple GPUs are supported but only on a single GPU node.