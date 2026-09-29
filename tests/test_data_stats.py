import numpy as np

from deepmd_jax.data import Dataset, _compute_stats_batch


def test_shared_stats_jit_matches_previous_result(tmp_path):
    set_dir = tmp_path / 'set.000'
    set_dir.mkdir()
    nframes = 4
    coords = np.zeros((nframes, 3, 3), dtype=np.float32)
    coords[:, :, 0] = [
        [0.0, 1.3, 2.7],
        [0.0, 1.4, 2.8],
        [0.0, 1.5, 2.9],
        [0.0, 1.6, 3.0],
    ]
    boxes = np.repeat(
        (np.eye(3, dtype=np.float32) * 6.0)[None], nframes, axis=0)
    np.savetxt(tmp_path / 'type.raw', [0, 0, 0], fmt='%d')
    np.save(set_dir / 'coord.npy', coords.reshape(nframes, -1))
    np.save(set_dir / 'box.npy', boxes.reshape(nframes, -1))
    np.save(set_dir / 'energy.npy', np.arange(nframes, dtype=np.float32))
    np.save(set_dir / 'force.npy', np.zeros((nframes, 9), dtype=np.float32))

    dataset = Dataset(
        str(tmp_path), ['coord', 'box', 'energy', 'force'])
    dataset.compute_lattice_candidate(2.5, False)
    dataset.pointer = 0
    _compute_stats_batch.clear_cache()
    stats = dataset.get_stats(2.5, 2)
    # Upstream dce47eb statistics for the fixed first two frames.
    np.testing.assert_allclose(stats['sr_mean'], [0.31026190519332886], rtol=0, atol=1e-8)
    np.testing.assert_allclose(stats['sr_std'], [0.029611550271511078], rtol=0, atol=1e-8)
    np.testing.assert_allclose(stats['Nnbrs'], 2.3333335, rtol=0, atol=1e-7)

    compiled = _compute_stats_batch._cache_size()
    assert compiled == 1
    other = Dataset(str(tmp_path), ['coord', 'box', 'energy', 'force'])
    other.compute_lattice_candidate(2.5, False)
    other.pointer = 0
    repeated = other.get_stats(2.5, 2)
    assert _compute_stats_batch._cache_size() == compiled
    for key in ['sr_mean', 'sr_std', 'Nnbrs']:
        np.testing.assert_array_equal(repeated[key], stats[key])
