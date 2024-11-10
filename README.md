# Deepmd-JAX

Welcome to Deepmd-JAX v0.2! Check this [example_notebook] for a quick tutorial. 
Supports Deep Potential (DP), Deep Wannier (DW), DP Long Range (DPLR), and you can try the Message Passing (DP-MP) model for more accuracy. 
You can perform NVE/NVT/NPT simulations on multiple GPUs, and enjoy a simple and lightning-fast training and simulation workflow!

# Installation
Requires CUDA 12 for GPU support; latest CUDA subversion preferred but not mandatory. We recommend using a conda environment
```bash
conda create -n deepmd-jax python=3.12
conda activate deepmd-jax
```
In your working directory, download and install with
```bash
git clone https://github.com/SparkyTruck/deepmd-jax.git
pip install -e ./deepmd-jax
```

