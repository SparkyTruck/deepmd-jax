# Deepmd-JAX

## Setting up environment on Perlmutter:
```bash
conda create --name deepmd-jax python=3.9
pip install ipykernel
python -m ipykernel install --user --name deepmd-jax
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax optax jax-md
```

## Tips for working on Perlmutter
- '''bash pip install gpustat''' to monitor GPU usage. Use '''bash hostname''' to check which login node you're at. Use nested ssh to jump to the desired node.
- VScode terminal sometimes does not load the '''bash module''' function. Use nested ssh to your node again to obtain a login shell.
 