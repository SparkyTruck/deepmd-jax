# Deepmd-JAX

## Setting up environment:
```bash
conda create --name deepmd-jax python=3.9
pip install ipykernel
python -m ipykernel install --user --name deepmd-jax
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax optax jax-md
pip install matplotlib
```
On HPCs, remember to activate the conda environment before submitting slurm jobs.

## Tips for working on Perlmutter
- ```pip install gpustat``` to monitor GPU usage. Use ```hostname``` to check which login node you're at. Use nested ssh to jump to the desired node.
- VScode terminal sometimes does not load the ```module``` function Use nested ssh to your node again to obtain a login shell.
 