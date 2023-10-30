# Deepmd-JAX

## Setting up environment:
```bash
conda create --name deepmd-jax python=3.10
conda activate deepmd-jax 
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax optax jax-md matplotlib
```
To run in Jupyter notebooks, create a kernel by
```
conda activate deepmd-jax 
pip install ipykernel
python -m ipykernel install --user --name deepmd-jax
```
