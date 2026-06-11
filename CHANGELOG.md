# Changelog

## 0.2.1

### Added
- Added path-integral molecular dynamics (PIMD) support through `Simulation(..., n_bead>1)` for NVT and NPT runs, including PILE-L Langevin thermostatting, bead-shaped positions/velocities, centroid reporting, and PIMD-aware neighbor handling.
- Added ASE-readable extended XYZ (`.xyz`/`.extxyz`) dataset support for training, testing, validation, and observable datasets, with chemical-type remapping and support for lists of extxyz files grouped by composition.
- Added support for testing datasets with variable atom counts by returning one result record per configuration.
- Added orthorhombic non-isotropic NPT controls through `couple_axes`, covering isotropic, semi-isotropic, uniaxial, and fully anisotropic orthorhombic box fluctuations.
- Added the `NVT_langevin` BAOAB Langevin routine with thermostat-work reporting.
- Added `atomic_t2` atom-wise tensor prediction support for symmetric 3x3 targets such as polarizability.

### Changed
- Reorganized simulation and dataset internals into smaller modules, including PIMD routines, simulation utilities, cached NPT routines, deprecated APIs, and generalized dataset loaders.
- Reworked trajectory dumping into `Simulation.run(...)` with `dump_prefix`, `dump_interval`, `dump_mode`, and `dump_content`; PIMD position dumps are now split by bead and include centroid dumps by default.
- Optimized XYZ dumping with persistent file handles, fewer unnecessary device-to-host copies, and compact float formatting matched to float precision.
- Updated `test()` to return `(error_metrics, test_results)`, where `error_metrics` contains `l1_mixed`, `rmse`, and `mae`, and `test_results` is a flat per-configuration list in dataset order.
- Updated `evaluate()` to return concatenated prediction arrays for energy/DPLR and atomic models.
- For DPLR testing/evaluation, reported energies and forces now include the explicit long-range contribution added back to the short-range model prediction.
- Changed the default training loss to `l1-mixed`; testing now reports `l1_mixed`, `rmse` (l2), and `mae` (l1).
- Moved atom type sorting into the neural-network model so dataset and simulation inputs can preserve user/input atom ordering.

### Deprecated
- Deprecated `TrajDump` and `TrajDumpSimulation`; use `Simulation.run(..., dump_prefix=..., dump_interval=...)` for streaming trajectory output.
