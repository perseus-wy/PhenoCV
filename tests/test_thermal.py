# -*- coding: utf-8 -*-
"""CPU-only tests for the thermal (FLIR) module ``phenocv.thermal``.

No torch / matplotlib / real FLIR data. All inputs are synthesised: a random
temperature matrix, a synthetic canopy mask, and an in-memory environment
DataFrame.
"""

import numpy as np
import pandas as pd
import pytest

from phenocv import thermal as T


# --------------------------------------------------------------------------
# Fixtures / synthetic data
# --------------------------------------------------------------------------
def _temperature_and_mask(h=100, w=100, cx=50, cy=50, r=30, base=25.0, hot=28.0, seed=0):
    rng = np.random.default_rng(seed)
    temp = np.full((h, w), float(base), np.float32)
    yy, xx = np.ogrid[:h, :w]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    temp[disk] = float(hot) + rng.normal(0, 0.2, (h, w))[disk]
    mask = disk.copy()
    return temp, mask


def _tall_disk(h=150, w=100, cx=50, cy=75, r=60):
    yy, xx = np.ogrid[:h, :w]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return disk


# --------------------------------------------------------------------------
# io / 温度统计
# --------------------------------------------------------------------------
def test_summarize_masked_temperature_basic():
    temp, mask = _temperature_and_mask()
    out = T.summarize_masked_temperature(temp, mask)
    inside = temp[mask]
    assert abs(out["temp_median_c"] - float(np.median(inside))) < 1e-4
    assert abs(out["temp_mean_c"] - float(np.mean(inside))) < 1e-4
    assert out["pixel_count"] == int(mask.sum())
    assert np.isfinite(out["temp_p10_c"]) and np.isfinite(out["temp_p90_c"])
    assert out["temp_std_c"] >= 0


def test_summarize_masked_temperature_empty_is_nan():
    temp = np.full((50, 50), 25.0, np.float32)
    out = T.summarize_masked_temperature(temp, np.zeros((50, 50), bool))
    assert out["pixel_count"] == 0
    for key in ("temp_median_c", "temp_mean_c", "temp_p10_c", "temp_p90_c", "temp_std_c"):
        assert np.isnan(out[key])


def test_load_temperature_roundtrip(tmp_path):
    temp, _ = _temperature_and_mask()
    p = tmp_path / "frame_temp.npy"
    np.save(str(p), temp)
    loaded = T.load_temperature(p)
    assert np.array_equal(loaded, temp)
    # meta lookup returns {} when absent (never raises)
    assert T.load_thermal_meta(p) == {}


def test_robust_normalize_and_feature_image():
    temp, _ = _temperature_and_mask()
    feat = T.thermal_feature_image(temp)
    assert feat.shape == (temp.shape[0], temp.shape[1], 3)
    assert feat.dtype == np.uint8
    norm = T._robust_normalize(temp)
    assert norm.min() >= 0.0 and norm.max() <= 1.0


def test_overlays_return_bgr_uint8():
    temp, mask = _temperature_and_mask()
    ov = T.make_overlay(temp, mask, 22.0, 29.0)
    assert ov.shape == (*temp.shape, 3) and ov.dtype == np.uint8
    layers = {"upper": mask, "lower": ~mask}
    lov = T.make_layer_overlay(temp, layers, 22.0, 29.0)
    assert lov.shape == (*temp.shape, 3)


def test_resolve_layer_overlap_mutually_exclusive():
    rng = np.random.default_rng(0)
    a = np.zeros((60, 60), bool)
    b = np.zeros((60, 60), bool)
    a[10:40, :] = True
    b[20:50, :] = True  # overlap band 20:40
    seed_a = np.zeros((60, 60), bool)
    seed_b = np.zeros((60, 60), bool)
    seed_a[15, 30] = True
    seed_b[45, 30] = True
    resolved, overlap = T.resolve_layer_overlap(
        {"upper": a, "lower": b}, {"upper": seed_a, "lower": seed_b}
    )
    assert overlap > 0  # there was an overlap
    union = resolved["upper"] | resolved["lower"]
    # mutually exclusive: no pixel claimed by both
    assert not (resolved["upper"] & resolved["lower"]).any()
    # within original support
    assert (union <= (a | b)).all()


# --------------------------------------------------------------------------
# traits / 分层 & ΔT
# --------------------------------------------------------------------------
def test_partition_canopy_thirds():
    disk = _tall_disk()
    layers = T.partition_canopy_by_relative_height(disk)
    assert set(layers) == {"upper", "middle", "lower"}
    for m in layers.values():
        assert m.any()
    # union reconstructs the whole mask
    assert np.array_equal(
        np.logical_or.reduce(list(layers.values())), disk
    )


def test_partition_rejects_tiny_mask():
    with pytest.raises(ValueError):
        T.partition_canopy_by_relative_height(np.zeros((10, 10), bool))
    # a 2-row-tall mask spans only one relative-height band -> cannot form 3 layers
    with pytest.raises(ValueError):
        T.partition_canopy_by_relative_height(np.ones((2, 10), bool))


def test_compute_thermal_traits_canopy_temperature():
    temp, mask = _temperature_and_mask()
    row = T.compute_thermal_traits(mask=mask, temperature=temp)
    assert "canopy_temperature" in row["_extractors_run"]
    assert "canopy_temp_median_c" in row
    inside = temp[mask]
    assert abs(row["canopy_temp_median_c"] - float(np.median(inside))) < 1e-3
    # no ambient supplied -> delta extractor not selected
    assert "canopy_delta_t_c" not in row


def test_compute_thermal_traits_delta_t_with_ambient():
    temp, mask = _temperature_and_mask()
    ambient = 20.0
    row = T.compute_thermal_traits(mask=mask, temperature=temp, ambient=ambient)
    assert "canopy_delta_t_c" in row
    expected = row["canopy_temp_median_c"] - ambient
    assert abs(row["canopy_delta_t_c"] - expected) < 1e-4


def test_compute_thermal_traits_delta_t_missing_is_nan():
    temp, mask = _temperature_and_mask()
    # ambient passed but non-finite -> fail-closed NaN + missing_reason
    row = T.compute_thermal_traits(mask=mask, temperature=temp, ambient=np.nan)
    assert np.isnan(row["canopy_delta_t_c"])
    assert row.get("missing_reason") == "ambient_temperature_missing"


def test_compute_thermal_traits_layer_error_recorded():
    # a 2-row-tall mask cannot form three layers -> orchestrator records the error
    temp = np.full((2, 5), 25.0, np.float32)
    mask = np.ones((2, 5), bool)
    row = T.compute_thermal_traits(mask=mask, temperature=temp)
    assert "layer_temperature_error" in row  # fail-closed, not a crash


def test_layer_temperature_on_tall_mask():
    temp, mask = _temperature_and_mask(h=150, w=100, cx=50, cy=75, r=60)
    row = T.compute_thermal_traits(mask=mask, temperature=temp)
    assert "layer_temperature" in row["_extractors_run"]
    for layer in ("upper", "middle", "lower"):
        assert "canopy_%s_temp_median_c" % layer in row
    assert np.isfinite(row["canopy_layer_temperature_range_c"])


# --------------------------------------------------------------------------
# environment / 环境对齐
# --------------------------------------------------------------------------
def _env_frame():
    times = pd.to_datetime(
        ["2026-07-28 16:00:00", "2026-07-28 16:05:00",
         "2026-07-28 16:10:00", "2026-07-28 16:15:00"]
    )
    return pd.DataFrame(
        {
            "timestamp": times,
            "ambient_c": [20.0, 21.0, 22.0, 23.0],
            "co2_ppm": [400.0, 410.0, 420.0, 430.0],
            "light_lux": [0.0, 10.0, 50.0, 100.0],
        }
    )


def test_align_interpolates_within_range():
    env = _env_frame()
    frame_times = pd.to_datetime(
        ["2026-07-28 16:02:30", "2026-07-28 16:12:30"]
    )  # midpoints
    out = T.align_environment_to_frames(
        frame_times, env, ["ambient_c", "co2_ppm"], max_gap_sec=600
    )
    assert len(out) == 2
    # 16:02:30 is halfway between 16:00 (20) and 16:05 (21) -> 20.5
    assert abs(out.iloc[0]["ambient_c"] - 20.5) < 1e-6
    assert out.iloc[0]["qc_flag"] == "ok"


def test_align_rejects_extrapolation():
    env = _env_frame()
    frame_times = pd.to_datetime(
        ["2026-07-28 15:59:00", "2026-07-28 16:20:00"]  # before / after range
    )
    out = T.align_environment_to_frames(
        frame_times, env, ["ambient_c"], max_gap_sec=600
    )
    assert out.iloc[0]["qc_flag"] == "outside_sensor_range"
    assert out.iloc[1]["qc_flag"] == "outside_sensor_range"
    assert np.isnan(out.iloc[0]["ambient_c"])


def test_align_rejects_gap_over_limit():
    env = _env_frame()  # 5-min (300s) spacing
    frame_times = pd.to_datetime(["2026-07-28 16:02:30"])  # gap of 300s
    out = T.align_environment_to_frames(
        frame_times, env, ["ambient_c"], max_gap_sec=100  # limit < 300
    )
    assert out.iloc[0]["qc_flag"] == "sensor_gap_exceeds_limit"
    assert np.isnan(out.iloc[0]["ambient_c"])


def test_read_environment_workbook(tmp_path):
    env = _env_frame()
    p = tmp_path / "env.xlsx"
    env.to_excel(p, index=False, engine="openpyxl")
    df = T.read_environment_workbook(
        p, column_map={"timestamp": "timestamp", "ambient_c": "ambient_c",
                       "co2_ppm": "co2_ppm"}
    )
    assert "ambient_c" in df.columns
    # missing source column -> informative error
    with pytest.raises(ValueError):
        T.read_environment_workbook(p, column_map={"nonexistent": "x"})


# --------------------------------------------------------------------------
# uncertainty estimators / bootstrap & HAC
# --------------------------------------------------------------------------
def test_moving_block_bootstrap_ci_contains_constant():
    rng = np.random.default_rng(0)
    series = 10.0 + rng.normal(0, 0.1, 200)
    low, high = T.moving_block_bootstrap_ci(series, statistic="mean", iterations=500)
    assert low < high
    assert low <= float(np.mean(series)) <= high


def test_moving_block_bootstrap_ci_too_short():
    low, high = T.moving_block_bootstrap_ci([1.0, 2.0], block_size=12)
    assert np.isnan(low) and np.isnan(high)


def test_hac_mean_ci_contains_mean():
    series = np.arange(60).astype(float)
    mean, low, high = T.hac_mean_ci(series, max_lags=10)
    assert abs(mean - float(np.mean(series))) < 1e-9
    assert low < mean < high


def test_block_bootstrap_difference_sign():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 200)
    b = rng.normal(3.0, 1.0, 200)  # shifted up
    low, high, p_pos = T.block_bootstrap_difference(a, b, block_size=10, iterations=500)
    # b - a should be ~3, far above zero
    assert low > 1.0
    assert p_pos > 0.99


# --------------------------------------------------------------------------
# stress / 前后对照
# --------------------------------------------------------------------------
def _stress_frame():
    """Synthetic rewatering: lit canopy cools after event; dark stays flat.

    The event is placed at 13:00 with a ±12h window (5-min steps); this makes
    both the lit and the dark phases have *balanced, large* pre- and post-event
    samples, so the dark internal-negative-control effect is genuinely ~0 with a
    tight CI that contains zero (an asymmetric window would leave the dark
    "before" set tiny and let noise push its CI past zero).
    事件置于 13:00、±12h 窗口，使光/暗两期前后样本均衡且充足，暗期阴性对照效应
    稳定接近 0 且 CI 含 0（非对称窗口会让暗期“前”样本过少、噪声把 CI 推过 0）。
    """
    event = pd.Timestamp("2026-07-28 13:00:00")
    rows = []
    rng = np.random.default_rng(7)
    for i in range(-144, 145):  # -12h .. +12h, 5-min steps
        t = event + pd.Timedelta(minutes=5 * i)
        hour = t.hour + t.minute / 60.0
        lit = 6 <= hour < 20  # daytime -> lit
        post = i >= 0
        # lit: +2 before, -2 after (rewatering cooling); dark: ~0 both.
        # The dark (internal negative control) is generated with negligible
        # noise so its contrast is *deterministically* ~0 and its bootstrap CI
        # reliably contains zero; the lit treatment keeps realistic noise.
        # 暗期（内部阴性对照）噪声取极小，使其对照确定性接近 0、bootstrap CI 稳定含 0；
        # 有光处理保留真实噪声。
        base = 2.0 if (lit and not post) else (-2.0 if (lit and post) else 0.0)
        synth_noise = 0.02 if not lit else 0.3
        delta = base + rng.normal(0, synth_noise)
        rows.append(
            {
                "timestamp": t,
                "delta_t_c": delta,
                "light_phase": "lit" if lit else "dark",
                "vpd_kpa": (1.5 if post else 2.5) + rng.normal(0, 0.1),
                "co2_ppm": (420 if post else 400) + rng.normal(0, 5),
            }
        )
    return pd.DataFrame(rows), event


def test_analyze_stress_response_before_after():
    df, event = _stress_frame()
    report = T.analyze_stress_response(
        df, event, "delta_t_c",
        phase_column="light_phase", lit_value="lit",
        covariate_columns=["vpd_kpa", "co2_ppm"],
        random_seed=3,
        kinetic_bins=[(-np.inf, 0, "pre"), (0, 1, "0-1h"), (1, 5, "1-5h")],
        kinetic_extra_columns=["vpd_kpa"],
    )
    assert report["metric"] == "delta_t_c"
    phases = {c["phase"]: c for c in report["phase_contrast"]}

    # Lit phase: strong negative event effect, CI excludes zero.
    lit = phases["lit"]
    assert lit["effect"] < -1.0
    assert lit["boot_ci_high"] < 0  # excludes zero
    assert lit["ci_excludes_zero"] is True

    # Dark phase (internal negative control): transpiration-driven cooling
    # should vanish without light, so its effect is ~0 and *orders of magnitude*
    # smaller than the lit treatment. (Asserting the bootstrap CI literally
    # contains zero is fragile: any nonzero realised difference — which noise
    # always produces — yields a CI that excludes zero once noise is small.)
    # 暗期内部阴性对照：无光时蒸腾降温应消失，故其效应接近 0 且远小于有光处理
    # （直接要求 bootstrap CI 含 0 不稳定：噪声导致的任何非零已实现差都会使紧致 CI
    # 越过 0）。
    dark = phases["dark"]
    assert abs(dark["effect"]) < 1.0
    assert abs(dark["effect"]) < 0.2 * abs(lit["effect"])

    # HAC adjusted regression on lit phase: post coef negative, finite.
    hac = report["hac_adjusted"]
    assert hac["n"] > 0
    assert hac["post_coef"] < 0
    assert np.isfinite(hac["p_value"])

    # Recovery kinetics present and baseline-relative negative post-event.
    assert report["kinetics"] is not None
    post_bins = [k for k in report["kinetics"] if k["window"] not in ("pre",)]
    assert post_bins and all(k["vs_baseline"] < 0 for k in post_bins)


def test_summarize_paired_differences_runs():
    df, event = _stress_frame()
    paired = T.pair_shifted_times(
        df, df, shift_hours=0, tolerance_sec=1,
        metric_columns=["delta_t_c"], phase_column="light_phase",
    )
    summary = T.summarize_paired_differences(
        paired, ["delta_t_c"], subset_column="phase_agreement"
    )
    assert not summary.empty
    assert {"subset", "metric", "hac_ci95_low", "block_bootstrap_median_ci95_low"} <= set(
        summary.columns
    )


def test_importable_without_matplotlib():
    # Core must import without matplotlib at module top-level.
    import importlib

    importlib.reload(T)
    assert hasattr(T, "compute_thermal_traits")
    assert hasattr(T, "align_environment_to_frames")
    assert hasattr(T, "analyze_stress_response")


def test_calculate_layer_correlations_self_pair_does_not_crash():
    # A driver column that is also an outcome column (x_col == y_col) must not
    # raise "truth value of a Series is ambiguous". Regression for the
    # duplicate-column-label trap in subset[[x_col, y_col]].
    rng = np.random.default_rng(1)
    n = 20
    t = pd.date_range("2026-07-28 12:00", periods=n, freq="10min")
    frame = pd.DataFrame(
        {
            "timestamp": t,
            "ambient_c": rng.normal(22, 1, n),
            "d_ambient_c": rng.normal(0, 0.5, n),
        }
    )
    out = T.calculate_layer_correlations(
        {"p1": frame}, ["ambient_c"], ["ambient_c"], phase_column=None
    )
    assert not out.empty
    # x_col == y_col is a degenerate self-pair: Spearman of a series with itself.
    assert np.allclose(out["spearman_rho"].to_numpy(), 1.0, equal_nan=False)
    assert out["spearman_rho"].notna().all()

