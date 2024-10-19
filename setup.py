import subprocess
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

setup(
    name='deepmd_jax',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'jax[cuda12]',
        'flax',
        'optax',
        'jax-md @ git+https://github.com/google/jax-md.git',
        'ase',
        'matplotlib',
        'gpustat',
        'ipykernel',
    ],
    author='Ruiqi Gao',
    author_email='ruiqigao@princeton.edu',
    description='DP-JAX',
    url='https://github.com/SparkyTruck/deepmd-jax',
    python_requires='>=3.10',
)
