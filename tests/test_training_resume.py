import json
import hashlib
import pickle
import shutil

import jax
import numpy as np
import pytest

from deepmd_jax.train import test as evaluate_model
from deepmd_jax.train import _load_portable_model_params, train
from deepmd_jax.data import Dataset
from deepmd_jax.dpmodel import DPModel
from deepmd_jax.utils import load_model


def _write_dataset(path):
    set_dir = path / 'set.000'
    set_dir.mkdir(parents=True)
    nframes = 8
    coords = np.zeros((nframes, 3, 3), dtype=np.float32)
    forces = np.zeros_like(coords)
    energies = np.zeros(nframes, dtype=np.float32)
    for frame in range(nframes):
        x1 = 1.25 + 0.025 * frame
        x2 = 2.70 - 0.015 * frame
        coords[frame, :, 0] = [0.0, x1, x2]
        d01 = x1 - 1.4
        d12 = (x2 - x1) - 1.4
        energies[frame] = 0.5 * (d01**2 + d12**2)
        forces[frame, 0, 0] = d01
        forces[frame, 1, 0] = -d01 + d12
        forces[frame, 2, 0] = -d12
    boxes = np.repeat(np.eye(3, dtype=np.float32)[None] * 6.0,
                      nframes, axis=0)
    np.savetxt(path / 'type.raw', np.zeros(3, dtype=int), fmt='%d')
    np.save(set_dir / 'coord.npy', coords.reshape(nframes, -1))
    np.save(set_dir / 'box.npy', boxes.reshape(nframes, -1))
    np.save(set_dir / 'energy.npy', energies)
    np.save(set_dir / 'force.npy', forces.reshape(nframes, -1))


def _train_kwargs(dataset, save_path, checkpoint_path, history_path):
    return dict(
        model_type='energy',
        rcut=2.5,
        train_data_path=str(dataset),
        val_data_path=str(dataset),
        save_path=str(save_path),
        checkpoint_path=str(checkpoint_path),
        history_path=str(history_path),
        checkpoint_every=1,
        step=6,
        seed=20260901,
        mp=False,
        embed_widths=[4, 4, 8],
        fit_widths=[8, 8],
        axis_neurons=2,
        batch_size=2,
        val_batch_size_ratio=1,
        print_every=2,
        getstat_bs=2,
        compress=False,
        loss='l2',
    )


def _write_portable_energy_params(dataset, path):
    train_data = Dataset(
        str(dataset), ['coord', 'box', 'force', 'energy'],
        {'atomic_sel': None, 'atomic_scalar': False},
        rng=np.random.default_rng(np.random.SeedSequence([20260901, 0])))
    train_data.compute_lattice_candidate(2.5, True, False)
    params = {
        'type': 'energy',
        'atomic_data_prefix': None,
        'embed_widths': [4, 4, 8],
        'embedMP_widths': None,
        'fit_widths': [8, 8],
        'axis': 2,
        'Ebias': train_data.fit_energy(),
        'rcut': 2.5,
        'use_2nd': True,
        'use_mp': False,
        'atomic': False,
        'hybrid': False,
        'nsel': None,
        'out_norm': 1.0,
        **train_data.get_stats(2.5, 2),
    }
    portable_raw = pickle.dumps(
        jax.tree_util.tree_map(
            lambda value: np.asarray(value) if hasattr(value, 'shape') else value,
            DPModel(params).params),
        protocol=pickle.HIGHEST_PROTOCOL)
    path.write_bytes(portable_raw)
    path.with_name(path.name + '.sha256').write_text(
        hashlib.sha256(portable_raw).hexdigest() + f'  {path.name}\n')
    return portable_raw


def _assert_trees_identical(left, right):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _write_pickle_with_sidecar(path, payload):
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    path.with_name(path.name + '.sha256').write_text(
        digest + f'  {path.name}\n')
    return digest


def test_portable_params_accept_measured_cpu_gpu_stat_drift(tmp_path):
    supplied = {
        'rcut': 6.0,
        'Ebias': np.array([-0.7870826721191406], dtype=np.float32),
        'sr_mean': np.array([0.06442106213018153], dtype=np.float64),
        'sr_std': np.array([0.09724441226940518], dtype=np.float64),
        'Nnbrs': np.float64(147.59352203147796),
    }
    computed = {
        'rcut': 6.0,
        'Ebias': supplied['Ebias'].copy(),
        'sr_mean': np.array([0.06442605329597993], dtype=np.float64),
        'sr_std': np.array([0.09724652347325068], dtype=np.float64),
        'Nnbrs': np.float64(147.57904837531368),
    }
    path = tmp_path / 'portable.pkl'
    digest = _write_pickle_with_sidecar(path, supplied)
    loaded, loaded_digest = _load_portable_model_params(path, computed)
    _assert_trees_identical(loaded, supplied)
    assert loaded_digest == digest


def test_portable_params_reject_larger_stat_or_contract_drift(tmp_path):
    supplied = {
        'rcut': 6.0,
        'Ebias': np.array([1.0], dtype=np.float32),
        'sr_mean': np.array([0.06], dtype=np.float64),
        'sr_std': np.array([0.10], dtype=np.float64),
        'Nnbrs': np.float64(150.0),
    }
    path = tmp_path / 'portable.pkl'
    _write_pickle_with_sidecar(path, supplied)
    bad_stats = dict(supplied, sr_mean=np.array([0.061], dtype=np.float64))
    with pytest.raises(ValueError, match='local data statistics: sr_mean'):
        _load_portable_model_params(path, bad_stats)
    bad_contract = dict(supplied, rcut=7.5)
    with pytest.raises(ValueError, match='model contract: rcut'):
        _load_portable_model_params(path, bad_contract)


def test_segment_resume_matches_continuous_training(tmp_path):
    dataset = tmp_path / 'dataset'
    _write_dataset(dataset)

    continuous = _train_kwargs(
        dataset, tmp_path / 'continuous.pkl', tmp_path / 'continuous.train.pkl',
        tmp_path / 'continuous.history.json')
    continuous_result = train(**continuous)
    assert continuous_result['completed']
    portable_params = tmp_path / 'portable-model-params.pkl'
    portable_raw = _write_portable_energy_params(dataset, portable_params)

    segmented = _train_kwargs(
        dataset, tmp_path / 'segmented.pkl', tmp_path / 'segmented.train.pkl',
        tmp_path / 'segmented.history.json')
    first_result = train(**segmented, max_updates_per_run=3)
    assert first_result == {
        'completed': False,
        'completed_updates': 3,
        'target_updates': 6,
        'checkpoint_path': str(tmp_path / 'segmented.train.pkl'),
    }
    assert not (tmp_path / 'segmented.pkl').exists()
    with open(segmented['checkpoint_path'], 'rb') as file:
        checkpoint = pickle.load(file)
    assert (hashlib.sha256(portable_raw).hexdigest()
            == checkpoint['contract']['model_params_sha256'])
    resumed_result = train(
        **segmented, resume=True, model_params_path=str(portable_params))
    assert resumed_result['completed']

    continuous_model, continuous_variables = load_model(
        continuous['save_path'], replicate=False)
    segmented_model, segmented_variables = load_model(
        segmented['save_path'], replicate=False)
    assert continuous_model.params == segmented_model.params
    _assert_trees_identical(continuous_variables, segmented_variables)
    continuous_history = json.loads(
        (tmp_path / 'continuous.history.json').read_text())
    segmented_history = json.loads(
        (tmp_path / 'segmented.history.json').read_text())
    assert continuous_history == segmented_history
    assert [record['update'] for record in continuous_history] == [2, 4, 6]


def test_checkpoint_hash_and_contract_fail_closed(tmp_path):
    dataset = tmp_path / 'dataset'
    _write_dataset(dataset)
    kwargs = _train_kwargs(
        dataset, tmp_path / 'model.pkl', tmp_path / 'model.train.pkl',
        tmp_path / 'history.json')
    result = train(**kwargs, max_updates_per_run=2)
    assert not result['completed']

    clean_checkpoint = tmp_path / 'clean.train.pkl'
    clean_sidecar = tmp_path / 'clean.train.pkl.sha256'
    shutil.copy2(kwargs['checkpoint_path'], clean_checkpoint)
    shutil.copy2(kwargs['checkpoint_path'] + '.sha256', clean_sidecar)

    with open(kwargs['checkpoint_path'], 'ab') as file:
        file.write(b'corruption')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        train(**kwargs, resume=True)

    contract_kwargs = dict(kwargs)
    contract_kwargs['checkpoint_path'] = str(clean_checkpoint)
    contract_kwargs['lr'] = 0.003
    with pytest.raises(ValueError, match='differing_fields=lr'):
        train(**contract_kwargs, resume=True)


def test_dpmp_train_save_reload_and_test(tmp_path):
    dataset = tmp_path / 'dataset'
    _write_dataset(dataset)
    model_path = tmp_path / 'dpmp.pkl'
    result = train(
        model_type='energy',
        rcut=2.5,
        train_data_path=str(dataset),
        save_path=str(model_path),
        step=2,
        seed=20260902,
        mp=True,
        embed_widths=[4, 4, 8],
        embed_mp_widths=[8, 8, 8],
        fit_widths=[8, 8],
        axis_neurons=2,
        batch_size=2,
        print_every=1,
        getstat_bs=2,
        compress=False,
        loss='l2',
    )
    assert result['completed']
    assert model_path.is_file()
    assert (tmp_path / 'dpmp.pkl.sha256').is_file()
    metrics, predictions = evaluate_model(str(model_path), str(dataset), batch_size=2)
    assert len(predictions) == 8
    assert np.isfinite(metrics['rmse']['energy'])
    assert np.isfinite(metrics['rmse']['force'])
    with open(model_path, 'ab') as file:
        file.write(b'corruption')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        load_model(str(model_path), replicate=False)
