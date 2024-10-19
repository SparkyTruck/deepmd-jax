# Deepmd-JAX

Supports Deep Potential, Deep Wannier, and Long Range (DPLR) models. Also, try the Deep Potential Message Passing (DP-MP) model for more accuracy. Experience blazing fast training and simulation of machine learning potentials!

# Installation
Requires CUDA>=12.0 to run on GPUs. We recommend using a conda environment
```bash
conda create -n deepmd-jax python=3.12
conda activate deepmd-jax
```
In your working directory, download and install with
```bash
git clone https://github.com/SparkyTruck/deepmd-jax.git
pip install -e ./deepmd-jax
```

## Quick Start Guide
Check this [example_notebook] for a quick start of training a Deep Potential model and running simulations with it. For more information, please refer to the [documentation].
