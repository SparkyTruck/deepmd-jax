# Deepmd-JAX

Welcome to **Deepmd-JAX v0.2**! For a quick tutorial, check out this [example_notebook].

## Supported Models
Deepmd-JAX supports:
- **Deep Potential (DP)**: Fast energy and force predictions.
- **Deep Wannier (DW)**: Predicting Wannier centers associated to atoms.
- **DP Long Range (DPLR)**: Incorporate explicit long-range Coulomb interactions.

Also, you can try the **DP-MP** architecture for enhanced accuracy.

Deepmd-JAX allows **NVE/NVT/NPT simulations** on multiple GPUs, offering a simple, lightning-fast training and simulation workflow.

## Installation

Requires **CUDA 12** for GPU support (latest CUDA subversion preferred, but not mandatory). We recommend using a conda environment:

`conda create -n deepmd-jax python=3.12`  
`conda activate deepmd-jax`

Clone the repository and install it in your working directory:

`git clone https://github.com/SparkyTruck/deepmd-jax.git`  
`pip install -e ./deepmd-jax`