from setuptools import setup, find_packages

setup(
    name='deepmd_jax',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'jax>=0.4.19',
        'flax', 'optax', 'jax-md', 'ase', 'jaxopt', 'matplotlib', 'gpustat'
    ],
    author='Ruiqi Gao',
    author_email='ruiqigao@princeton.edu',
    description='DP-JAX',
    url='https://github.com/SparkyTruck/deepmd-jax',
    python_requires='>=3.9',
)
