import numpy as np
import pytest

from cardiostream.signal import (
    FS,
    WINDOW,
    CausalFilter,
    beat_window,
    detect_qrs,
    filter_record,
    match_peaks,
    rr_feature_matrix,
)


def synthetic_ecg(r_peaks, n, seed=0):
    """Narrow Gaussian 'QRS' spikes plus a small T wave, baseline wander and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = 0.1 * np.sin(2 * np.pi * 0.3 * t / FS) + 0.02 * rng.standard_normal(n)
    for r in r_peaks:
        x += 1.2 * np.exp(-0.5 * ((t - r) / 4) ** 2)
        x += 0.25 * np.exp(-0.5 * ((t - r - 90) / 20) ** 2)
    return x


def regular_peaks(n_beats=60, rr=0.8, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    gaps = rr * FS * (1 + jitter * rng.standard_normal(n_beats))
    return (FS + np.cumsum(gaps)).astype(int)


def test_chunked_filter_matches_whole_record():
    x = np.random.default_rng(1).standard_normal(5000)
    whole = filter_record(x)
    f = CausalFilter()
    chunks = np.concatenate([f(x[i : i + 37]) for i in range(0, len(x), 37)])
    np.testing.assert_allclose(chunks, whole, atol=1e-10)


@pytest.mark.parametrize("chunk", [1, 14, 360, 5000])
def test_detector_finds_beats_regardless_of_chunk_size(chunk):
    peaks = regular_peaks(jitter=0.08)
    x = filter_record(synthetic_ecg(peaks, peaks[-1] + 2 * FS))
    det = detect_qrs(x, chunk=chunk)
    pairs, fp, fn = match_peaks(det, peaks)
    # The first ~2 s are used to learn signal levels; after that, every beat.
    assert len(fp) == 0
    assert len(fn) <= 3
    # R location is accurate to within ~15 ms (filter group delay included).
    errs = [abs(det[i] - peaks[j]) for i, j in pairs]
    assert np.median(errs) <= 0.015 * FS


def test_detector_output_identical_across_chunkings():
    peaks = regular_peaks(jitter=0.1, seed=3)
    x = filter_record(synthetic_ecg(peaks, peaks[-1] + 2 * FS, seed=3))
    np.testing.assert_array_equal(detect_qrs(x, chunk=7), detect_qrs(x, chunk=900))


def test_detector_is_silent_on_flat_noise():
    x = filter_record(0.01 * np.random.default_rng(0).standard_normal(20 * FS))
    assert len(detect_qrs(x)) < 30  # noise must not look like a 90+ bpm rhythm


def test_match_peaks_tolerance():
    pairs, fp, fn = match_peaks(np.array([100, 500, 2000]), np.array([110, 700, 2000]))
    assert pairs == [(0, 0), (2, 2)]
    assert list(fp) == [1] and list(fn) == [1]


def test_rr_features_flag_premature_beat():
    r = np.array([0, 288, 576, 864, 1000, 1440, 1728])  # beat at 1000 is early
    feats, idx = rr_feature_matrix(r)
    assert list(idx) == [1, 2, 3, 4, 5]
    early = feats[list(idx).index(4)]
    pre_ratio, post_ratio = early[3], early[4]
    assert pre_ratio < 0.6 and post_ratio > 1.3


def test_beat_window_shape_and_edges():
    x = np.sin(np.arange(2000) / 10)
    w = beat_window(x, 1000)
    assert w.shape == (WINDOW,) and w.dtype == np.float32
    assert abs(float(w.mean())) < 0.5 and abs(float(w.std()) - 1) < 1e-3
    assert beat_window(x, 10) is None
    assert beat_window(x, 1990) is None
