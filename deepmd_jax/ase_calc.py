"""ASE Calculator interface to deepmd-jax models.

Split out of md.py. Importing this module pulls in ASE and jax_md; keep it out
of the training path.
"""
import jax
import jax.numpy as jnp
import jax_md
import numpy as np
import flax.linen as nn
from jax.sharding import PartitionSpec as PSpec
from ase.calculators.calculator import Calculator, all_changes

from .data import compute_lattice_candidate
from .utils import load_model


class DPJaxCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self,
            model_path,
            type_idx = None,
            dtype=jnp.float32,
            **kwargs):

        self.atoms = None
        self.use_cache = False
        self._dtype = dtype

        self._model, self._variables = load_model(model_path)
        self._static_args = None

        type_idx_np = np.array(type_idx).astype(int)
        ct = self._model.params.get('chemical_types')
        if ct is not None:
            unknown = set(type_idx_np.tolist()) - set(ct)
            if unknown:
                raise ValueError('Atomic numbers %s in type_idx are not in model.params["chemical_types"]=%s'
                                 % (sorted(unknown), ct))
            z_to_idx = {z: i for i, z in enumerate(ct)}
            type_idx_np = np.array([z_to_idx[z] for z in type_idx_np], dtype=int)
        self._type_idx = type_idx_np
        type_count = np.bincount(self._type_idx)
        self._type_count = np.pad(type_count, (0, self._model.params['ntypes'] - len(type_count)))

        self._energy_and_forces_fn = self._get_energy_and_forces_fn()
        print("# Initializing the DPJaxCalculator")

    def _get_energy_and_forces_fn(self, model_and_variables=None):
        if model_and_variables is None:
            model_and_variables = (self._model, self._variables)
        model, variables = model_and_variables

        def energy_fn(coord, box, static_args, nbrs_nm=None, perturbation=None, **kwargs):
            '''
                You can customize the energy function here, i.e. if you want to add perturbations.
            '''
            # perturbation = 1, required by jax-md stress calculation
            if perturbation is not None:
                coord = coord @ perturbation
                box = box @ perturbation
            # Ensure coord and box is replicated on all devices
            if len(jax.devices()) > 1:
                coord = jax.lax.with_sharding_constraint(coord, PSpec())
                box = jax.lax.with_sharding_constraint(box, PSpec())

            # Energy calculation
            E = model.apply(variables,
                            coord,
                            box,
                            static_args,
                            nbrs_nm)[0]
            if model.params['type'] == 'dplr':
                raise NotImplementedError("DPLR model not implemented in the ASE calculator yet.")
            return E

        def stress_fn(coord, box, static_args, **kwargs):
            return jax_md.quantity.stress(
                        energy_fn,
                        coord,
                        box,
                        static_args=static_args,
                        velocity=None,
                        nbrs_nm=None,
                    )

        def e_and_f_and_s(coords, box, static_args, **kwargs):
            e, grad = jax.value_and_grad(energy_fn)(coords, box, static_args, **kwargs)
            stress = stress_fn(coords, box, static_args, **kwargs)
            stress_voigt = jnp.array([
                stress[0, 0],
                stress[1, 1],
                stress[2, 2],
                stress[1, 2],
                stress[0, 2],
                stress[0, 1],
                ], dtype=self._dtype)
            # Note the minus sign in the stress below, to match ASE's convention
            # Also, the off-diagonal components have not been tested
            return e, -grad, -stress_voigt

        return jax.jit(e_and_f_and_s, static_argnames=('static_args',))


    def _get_static_args(self, position):
        '''
            Returns a FrozenDict of the complete set of static arguments for jit compilation.
        '''
        box = self._current_box
        # it is important to disable_ortho to compute the stress tensor correctly, even for orthogonal boxes
        lattice_args = compute_lattice_candidate(box[None], self._model.params['rcut'], print_info=False, disable_ortho=True)
        static_args = nn.FrozenDict({'type_idx':tuple(self._type_idx), 'lattice':lattice_args, 'use_neighbor_list':False})
        return static_args

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):

        if atoms is not None:
            self.atoms = atoms.copy()

        cell = np.asarray(self.atoms.get_cell(complete=True))  # Use complete=True for (3,3)
        box = jnp.array(cell, dtype=self._dtype)
        self._current_box = box

        # Get positions and cell from ASE
        pos = self.atoms.get_positions()  # (N,3), in Å
        self._natoms = pos.shape[0]
        coords = jnp.array(pos, dtype=self._dtype)

        static_args = self._get_static_args(coords)
        if self._static_args != static_args:
            print('# Lattice vectors for neighbor images: Max %d out of %d candidates.' % (static_args['lattice']['lattice_max'], len(static_args['lattice']['lattice_cand'])))
        self._static_args = static_args

        E, F, S = self._energy_and_forces_fn(
            coords,
            box,
            static_args,
            )

        # Convert JAX arrays to numpy
        self.results["energy"] = float(E)
        self.results["forces"] = np.asarray(F)
        self.results["stress"] = np.asarray(S)
