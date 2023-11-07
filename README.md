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
- `examples/train.py`: Training a model; you should prepare your data in the format of DeepMD-kit.
- `examples/evaluate.ipynb`: Evaluating model predictions
- `examples/simulate.py`: Simulating a system in NVT/NVE ensemble with trained models
- `examples/sub` and `examples/sub_simulation`: Example slurms scripts for running on clusters (here it is Perlmutter@NERSC).

You can modify the paramters inside the the scripts to fit your needs, or create you're own script from it.

Note: In training, no neighbor list need to be used. In simulations, neighbor lists are only supported when the periodic box is orthorhombic and the model cutoff is less than half of the box length. 

Note: In the script `examples/simulate.py`, multiple GPU/TPU devices can be used for simulation with neighbor lists, but it only runs on a single compute node.