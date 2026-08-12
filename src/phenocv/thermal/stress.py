# -*- coding: utf-8 -*-
"""
Stress / rewatering response analysis + generic uncertainty estimators.

This module generalises the private FLIR rewatering study into a reusable
toolkit. It deliberately drops every dataset-specific identifier (plant-id-like
tokens such as ``label_2..7``, example sensor serials such as ``SENSOR_001`` /
``SENSOR_002``, network-share paths, checkpoint paths) and operates on
**plain DataFrames with documented column conventions**.

Two uncertainty estimators are exported as standalone, reusable tools:

  * :func:`moving_block_bootstrap_ci` — circular moving-block bootstrap 95% CI
    for a time-series statistic (preserves autocorrelation structure).
  * :func:`hac_mean_ci` — Newey–West (HAC) robust 95% CI for the mean.

The entry point :func:`analyze_stress_response` performs a before/after
contrast around an event (e.g. irrigation / rewatering), reports the effect
with a block-bootstrap CI, runs a covariate-adjusted HAC regression (the
internal light/dark negative control), and optionally tabulates recovery
kinetics. Supporting generic roll-ups (``summarize_plants``,
``summarize_paired_differences``, ``detect_light_transitions``,
``calculate_layer_correlations``, ``pair_shifted_times``) let callers build
population-level comparisons without any private hard-coding.

胁迫 / 复水响应分析与通用不确定性估计工具。

本模块把私有的 FLIR 复水研究泛化为可复用工具箱，去除一切数据集特定标识
（plant-id 类 token、示例传感器序列号、网络共享/checkpoint 路径），仅对约定列名的普通 DataFrame
操作。导出两种不确定性估计器（移动块 bootstrap / HAC 均值 CI）与通用前后对照入口
``analyze_stress_response``，并提供若干通用汇总函数，供构建人群级比较而无需任何
私人硬编码。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Physical helper / 物理辅助
# --------------------------------------------------------------------------
def vapour_pressure_deficit(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """Tetens-formula saturation vapour-pressure deficit (kPa).

    VPD = es(T) · (1 − RH/100), with es(T) = 0.6108·exp(17.27·T/(T+237.3)).
    用 Tetens 公式计算饱和水汽压差（kPa）。
    """
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    return es * (1.0 - rh_pct / 100.0)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR-adjusted q-values (monotonic from the end).

    Benjamini–Hochberg 错误发现率校正（自末尾单调化）。
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


# --------------------------------------------------------------------------
# Uncertainty estimators / 不确定性估计器
# --------------------------------------------------------------------------
def moving_block_bootstrap_ci(
    values: Sequence[float],
    statistic: str = "mean",
    iterations: int = 4000,
    block_size: int = 12,
    random_seed: int = 20260728,
) -> Tuple[float, float]:
    """Circular moving-block bootstrap 95% CI for a time-series statistic.

    Resamples whole blocks (preserving autocorrelation) and returns the
    2.5th / 97.5th percentile of the re-sampled statistic. Returns
    ``(nan, nan)`` when fewer than ``block_size`` finite values are available.

    循环移动块 bootstrap 的 95% 区间：整块重采样（保留自相关），返回统计量重采样
    分布的 2.5/97.5 分位。有限值少于 block_size 时返回 (nan, nan)。

    Parameters
    ----------
    values : array-like
    statistic : "mean" | "median"
    iterations : number of bootstrap replicates
    block_size : block length (≈ autocorrelation window)
    random_seed : RNG seed for reproducibility
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < block_size:
        return float("nan"), float("nan")
    generator = np.random.default_rng(random_seed)
    blocks_needed = int(np.ceil(len(array) / block_size))
    estimates = np.empty(iterations, dtype=float)
    offsets = np.arange(block_size)
    for iteration in range(iterations):
        starts = generator.integers(0, len(array), size=blocks_needed)
        indices = (starts[:, None] + offsets[None, :]) % len(array)
        sample = array[indices.ravel()[: len(array)]]
        estimates[iteration] = (
            np.median(sample) if statistic == "median" else np.mean(sample)
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def block_bootstrap_difference(
    a: Sequence[float],
    b: Sequence[float],
    block_size: int = 12,
    iterations: int = 4000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Circular moving-block bootstrap CI for the difference of two means.

    Estimates ``mean(b) − mean(a)`` while preserving each series' autocorrelation
    via whole-block resampling. Returns ``(low, high, p_positive)`` where
    ``p_positive`` is the fraction of bootstrap replicates with a positive
    difference (a one-sided extremity heuristic).

    两序列均值差的循环移动块 bootstrap 区间：整块重采样保留各自自相关，估计
    ``mean(b) − mean(a)``。返回 (低, 高, P(差>0))。
    """
    rng = np.random.default_rng(seed)
    xa = np.asarray(a, dtype=float)[np.isfinite(a)]
    xb = np.asarray(b, dtype=float)[np.isfinite(b)]

    def resample(x: np.ndarray) -> np.ndarray:
        n = len(x)
        n_blocks = int(np.ceil(n / block_size))
        starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
        return np.concatenate([x[s : s + block_size] for s in starts])[:n]

    diffs = np.array(
        [resample(xb).mean() - resample(xa).mean() for _ in range(iterations)]
    )
    return (
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
        float((diffs >= 0).mean()),
    )


def hac_mean_ci(
    values: Sequence[float],
    max_lags: int = 12,
) -> Tuple[float, float, float]:
    """Mean and Newey–West (HAC) 95% CI for an autocorrelated series.

    Fits an intercept-only OLS with a HAC covariance (lag selected as
    ``min(max_lags, n//4)``) and returns ``(mean, ci_low, ci_high)``. Statsmodels
    is imported lazily so ``import phenocv.thermal`` does not require it.

    对自相关序列估计均值与 Newey–West (HAC) 95% 区间。惰性导入 statsmodels，使
    ``import phenocv.thermal`` 本身不依赖该包。返回 (均值, 低, 高)。
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    import statsmodels.api as sm

    design = np.ones((len(array), 1), dtype=float)
    fitted = sm.OLS(array, design).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": min(max_lags, max(1, len(array) // 4))},
    )
    interval = fitted.conf_int(alpha=0.05)[0]
    return float(fitted.params[0]), float(interval[0]), float(interval[1])


# --------------------------------------------------------------------------
# Before/after contrast primitives / 前后对照原语
# --------------------------------------------------------------------------
def phase_contrast(
    before: Sequence[float],
    after: Sequence[float],
    block_size: int = 12,
    iterations: int = 4000,
    seed: int = 42,
) -> Dict[str, float]:
    """Before/after contrast for one phase: effect + block-bootstrap CI + t-test.

    单阶段前后对照：效应量 + 移动块 bootstrap 区间 + Welch t 检验 p 值。
    """
    a = np.asarray(before, dtype=float)[np.isfinite(before)]
    b = np.asarray(after, dtype=float)[np.isfinite(after)]
    out = {
        "n_before": int(a.size),
        "n_after": int(b.size),
        "mean_before": float(a.mean()) if a.size else float("nan"),
        "mean_after": float(b.mean()) if b.size else float("nan"),
        "effect": float(b.mean() - a.mean()) if a.size and b.size else float("nan"),
        "boot_ci_low": float("nan"),
        "boot_ci_high": float("nan"),
        "p_value": float("nan"),
        "ci_excludes_zero": False,
    }
    if a.size >= 2 and b.size >= 2:
        # block_bootstrap_difference(a, b) returns mean(b) - mean(a), i.e.
        # mean(after) - mean(before) when a=before, b=after.
        # block_bootstrap_difference(a, b) 返回 mean(b) − mean(a)，即
        # a=before、b=after 时为 mean(after) − mean(before)。
        lo, hi, p_pos = block_bootstrap_difference(a, b, block_size, iterations, seed)
        out["boot_ci_low"] = lo
        out["boot_ci_high"] = hi
        # CI excludes zero iff 0 is *outside* the closed interval [lo, hi].
        # 当 0 落在闭区间 [lo, hi] 之外时，CI 不含 0。
        out["ci_excludes_zero"] = (lo <= 0 <= hi) is False and not (
            np.isnan(lo) or np.isnan(hi)
        )
        from scipy import stats as _stats

        tstat, pval = _stats.ttest_ind(b, a, equal_var=False)
        out["p_value"] = float(pval)
    return out


def hac_adjusted_regression(
    df: pd.DataFrame,
    metric: str,
    covariate_columns: Optional[Sequence[str]] = None,
    max_lags: int = 12,
    post_column: str = "post",
) -> Dict[str, float]:
    """Covariate-adjusted OLS of ``metric ~ post + covariates`` with HAC errors.

    The ``post`` coefficient is the event effect, robust to autocorrelation.
    Returns the coefficient, HAC standard error, p-value, 95% CI, R² and a dict
    of covariate coefficients. Statsmodels is imported lazily.

    协变量校正的 OLS（HAC 稳健标准误）：``metric ~ post + 协变量``。``post`` 系数即
    事件效应（对自相关稳健）。惰性导入 statsmodels。
    """
    import statsmodels.api as sm

    cols = [post_column] + list(covariate_columns or [])
    sub = df[[metric, *cols]].dropna()
    if sub.empty:
        return {"n": 0, "post_coef": float("nan"), "hac_se": float("nan"),
                "p_value": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "r_squared": float("nan"),
                "covariate_coefs": {}}
    y = sub[metric]
    X = sm.add_constant(sub[cols])
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    ci = model.conf_int().loc[post_column]
    cov_coefs = {
        c: float(model.params[c])
        for c in cols
        if c != post_column
    }
    return {
        "n": int(len(sub)),
        "post_coef": float(model.params[post_column]),
        "hac_se": float(model.bse[post_column]),
        "p_value": float(model.pvalues[post_column]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "r_squared": float(model.rsquared),
        "covariate_coefs": cov_coefs,
    }


def recovery_kinetics(
    df: pd.DataFrame,
    metric: str,
    event_time,
    time_column: str = "timestamp",
    post_column: Optional[str] = None,
    bins: Optional[Sequence[Tuple[float, float, str]]] = None,
    baseline: Optional[float] = None,
    extra_columns: Optional[Sequence[str]] = None,
) -> List[Dict[str, float]]:
    """Tabulate post-event recovery: mean of ``metric`` per time-window (hours).

    复水恢复动力学：按事件后小时数窗口汇总 ``metric`` 均值（相对基线变化）。

    Parameters
    ----------
    bins : list of (lo_h, hi_h, name) windows in hours relative to ``event_time``.
    baseline : pre-event mean used for ``vs_baseline``; defaults to pre-event mean.
    """
    if bins is None:
        bins = [
            (-np.inf, 0, "pre_event_all"),
            (0, 1, "0-1h"),
            (1, 2, "1-2h"),
            (2, 4, "2-4h"),
            (4, 18, "4-18h"),
            (18, 25, "18-25h"),
        ]
    sub = df.copy()
    sub["_t"] = pd.to_datetime(sub[time_column])
    event = pd.Timestamp(event_time)
    sub["_hours"] = (sub["_t"] - event).dt.total_seconds() / 3600.0
    if post_column is None:
        sub["_post"] = (sub["_hours"] >= 0).astype(int)
    else:
        sub["_post"] = sub[post_column].astype(int)
    if baseline is None:
        pre = sub.loc[sub["_post"] == 0, metric]
        baseline = float(pre.mean()) if pre.notna().any() else float("nan")
    extras = list(extra_columns or [])
    rows: List[Dict[str, float]] = []
    for lo, hi, name in bins:
        win = sub[(sub["_hours"] > lo) & (sub["_hours"] <= hi)]
        if len(win) == 0:
            continue
        row: Dict[str, float] = {
            "window": name,
            "n": int(len(win)),
            "mean": float(win[metric].mean()),
            "vs_baseline": float(win[metric].mean() - baseline),
        }
        for ec in extras:
            if ec in win.columns:
                row[ec] = float(win[ec].mean())
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# High-level entry point / 高层入口
# --------------------------------------------------------------------------
def analyze_stress_response(
    timeseries_df: pd.DataFrame,
    event_time,
    metric: str,
    *,
    time_column: str = "timestamp",
    post_column: Optional[str] = None,
    phase_column: Optional[str] = None,
    lit_value: Any = None,
    covariate_columns: Optional[Sequence[str]] = None,
    block_size: int = 12,
    bootstrap_iterations: int = 4000,
    hac_max_lags: int = 12,
    random_seed: int = 42,
    kinetic_bins: Optional[Sequence[Tuple[float, float, str]]] = None,
    kinetic_extra_columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Before/after contrast around an event with uncertainty + light/dark control.

    Around ``event_time`` (e.g. irrigation / rewatering), contrast the chosen
    ``metric`` before vs after, per photoperiod phase (the dark phase is the
    internal negative control — transpiration-driven cooling should vanish
    without light). Reports:

      * ``phase_contrast`` — effect, block-bootstrap 95% CI and Welch p per phase.
      * ``hac_adjusted`` — covariate-adjusted OLS of ``metric ~ post`` (+ covariates),
        HAC-robust, run on the lit phase (or all if undetermined).
      * ``kinetics`` — optional post-event recovery bins (see :func:`recovery_kinetics`).

    围绕事件（如灌溉/复水）做前后对照，带不确定性估计与光暗内部阴性对照：
    暗期作为内部阴性对照（无光则蒸腾降温应消失）。返回 phase_contrast /
    hac_adjusted / kinetics。

    Parameters
    ----------
    timeseries_df : DataFrame with ``time_column`` and ``metric`` (plus optional
        ``phase_column`` for the light/dark split).
    event_time : event timestamp (the treatment instant). 事件时刻。
    metric : column to contrast. 对照的指标列。
    phase_column : categorical photoperiod column (e.g. "light_phase"). 光暗阶段列。
    lit_value : value of ``phase_column`` denoting the lit phase (for the HAC fit
        and the "lit" contrast label). 表示有光阶段的值。
    covariate_columns : covariates for the adjusted HAC regression (e.g. VPD,
        light, CO₂). HAC 回归的协变量。
    """
    df = timeseries_df.copy()
    df["_t"] = pd.to_datetime(df[time_column])
    event = pd.Timestamp(event_time)
    if post_column is None:
        df["_post"] = (df["_t"] >= event).astype(int)
    else:
        df["_post"] = df[post_column].astype(int)

    is_lit = None
    if phase_column is not None and lit_value is not None:
        is_lit = df[phase_column] == lit_value

    phases: List[Tuple[str, pd.DataFrame]] = [("all", df)]
    if phase_column is not None:
        for pv in df[phase_column].dropna().unique():
            phases.append((str(pv), df[df[phase_column] == pv]))

    contrast = []
    for phase_name, subset in phases:
        before = subset.loc[subset["_post"] == 0, metric]
        after = subset.loc[subset["_post"] == 1, metric]
        c = phase_contrast(
            before, after,
            block_size=block_size,
            iterations=bootstrap_iterations,
            seed=random_seed,
        )
        c["phase"] = phase_name
        contrast.append(c)

    # HAC adjusted regression on the lit phase if derivable, else all.
    hac_subset = df
    if is_lit is not None:
        lit_df = df[is_lit]
        hac_subset = lit_df if len(lit_df) >= 30 else df
    adjusted = hac_adjusted_regression(
        hac_subset, metric,
        covariate_columns=list(covariate_columns or []),
        max_lags=hac_max_lags,
        post_column="_post",
    )

    kinetics = None
    if kinetic_bins is not None:
        kinetics = recovery_kinetics(
            df, metric, event, time_column=time_column,
            post_column="_post", bins=kinetic_bins,
            extra_columns=list(kinetic_extra_columns or []),
        )

    return {
        "event_time": str(event),
        "metric": metric,
        "n_total": int(len(df)),
        "phase_contrast": contrast,
        "hac_adjusted": adjusted,
        "kinetics": kinetics,
    }


# --------------------------------------------------------------------------
# Generic population roll-ups / 通用人群级汇总
# --------------------------------------------------------------------------
def summarize_plants(
    frames: Dict[str, pd.DataFrame],
    metric_columns: Sequence[str],
    phase_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-plant and per-phase summaries over a dict of {plant_id: DataFrame}.

    对 {plant_id: DataFrame} 字典生成整株与光暗阶段摘要。
    """
    metrics = list(metric_columns)
    plant_rows: List[Dict[str, Any]] = []
    phase_rows: List[Dict[str, Any]] = []
    for plant_id, frame in frames.items():
        time_col = next(
            (c for c in ("timestamp", "timestamp_local") if c in frame.columns),
            frame.columns[0] if len(frame.columns) else None,
        )
        plant_row: Dict[str, Any] = {
            "plant_id": plant_id,
            "n": int(len(frame)),
            "start_time": (
                pd.Timestamp(frame[time_col].iloc[0]).isoformat()
                if time_col is not None and not frame.empty
                else None
            ),
            "end_time": (
                pd.Timestamp(frame[time_col].iloc[-1]).isoformat()
                if time_col is not None and not frame.empty
                else None
            ),
        }
        for m in metrics:
            col = frame[m].dropna()
            plant_row["%s_min" % m] = float(col.min()) if col.size else float("nan")
            plant_row["%s_median" % m] = (
                float(col.median()) if col.size else float("nan")
            )
            plant_row["%s_max" % m] = float(col.max()) if col.size else float("nan")
        plant_rows.append(plant_row)

        if phase_column is not None and phase_column in frame.columns:
            for phase, subset in frame.groupby(phase_column, sort=False):
                phase_row: Dict[str, Any] = {
                    "plant_id": plant_id,
                    "light_phase": phase,
                    "n": int(len(subset)),
                }
                for m in metrics:
                    col = subset[m].dropna()
                    phase_row["%s_mean" % m] = (
                        float(col.mean()) if col.size else float("nan")
                    )
                    phase_row["%s_median" % m] = (
                        float(col.median()) if col.size else float("nan")
                    )
                    phase_row["%s_std" % m] = (
                        float(col.std()) if col.size else float("nan")
                    )
                phase_rows.append(phase_row)
    return pd.DataFrame(plant_rows), pd.DataFrame(phase_rows)


def summarize_paired_differences(
    paired: pd.DataFrame,
    metric_columns: Sequence[str],
    *,
    subset_column: Optional[str] = None,
    hac_max_lags: int = 12,
    bootstrap_iterations: int = 4000,
    block_size: int = 12,
    random_seed: int = 20260728,
) -> pd.DataFrame:
    """Summarise paired differences across subsets (e.g. photoperiod agreement).

    ``paired`` must contain ``diff__<metric>`` columns (produced by
    :func:`pair_shifted_times`). For each subset × metric it reports the HAC mean
    CI, the block-bootstrap median CI and the positive fraction.

    对配对差异做子集级汇总。``paired`` 须含 ``diff__<metric>`` 列（由
    :func:`pair_shifted_times` 生成）。对每个 子集×指标 报告 HAC 均值区间、块
    bootstrap 中位数区间与正值占比。
    """
    metrics = list(metric_columns)
    if subset_column is not None and subset_column in paired.columns:
        subsets = {
            str(v): paired[paired[subset_column] == v]
            for v in paired[subset_column].dropna().unique()
        }
    else:
        subsets = {"all": paired}

    rows: List[Dict[str, Any]] = []
    for subset_name, subset in subsets.items():
        for metric_index, metric in enumerate(metrics):
            column = "diff__%s" % metric
            if column not in subset.columns:
                continue
            values = subset[column].dropna()
            if values.size == 0:
                continue
            mean, mean_low, mean_high = hac_mean_ci(values, hac_max_lags)
            median_low, median_high = moving_block_bootstrap_ci(
                values,
                statistic="median",
                iterations=bootstrap_iterations,
                block_size=min(block_size, len(values)),
                random_seed=random_seed + metric_index,
            )
            rows.append(
                {
                    "subset": subset_name,
                    "metric": metric,
                    "n": int(len(values)),
                    "mean": mean,
                    "hac_ci95_low": mean_low,
                    "hac_ci95_high": mean_high,
                    "median": float(values.median()),
                    "block_bootstrap_median_ci95_low": median_low,
                    "block_bootstrap_median_ci95_high": median_high,
                    "positive_fraction": float(values.gt(0).mean()),
                }
            )
    return pd.DataFrame(rows)


def detect_light_transitions(
    frames: Dict[str, pd.DataFrame],
    timestamp_column: str,
    phase_column: str,
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    """Record photoperiod transitions per plant (helps interpret circadian rhythm).

    记录每株的光暗阶段转换（辅助解释昼夜节律）。
    """
    metrics = list(metric_columns)
    rows: List[Dict[str, Any]] = []
    for plant_id, frame in frames.items():
        if phase_column not in frame.columns or timestamp_column not in frame.columns:
            continue
        changed = frame[phase_column].ne(frame[phase_column].shift())
        for index in frame.index[changed]:
            row: Dict[str, Any] = {
                "plant_id": plant_id,
                "timestamp_local": pd.Timestamp(
                    frame.loc[index, timestamp_column]
                ).isoformat(),
                "clock_time": pd.Timestamp(
                    frame.loc[index, timestamp_column]
                ).strftime("%H:%M:%S"),
                "light_phase": frame.loc[index, phase_column],
            }
            for m in metrics:
                if m in frame.columns:
                    row[m] = float(frame.loc[index, m])
            rows.append(row)
    return pd.DataFrame(rows)


def calculate_layer_correlations(
    frames: Dict[str, pd.DataFrame],
    driver_columns: Sequence[str],
    outcome_columns: Sequence[str],
    phase_column: Optional[str] = None,
) -> pd.DataFrame:
    """Spearman correlations between environment drivers and layer outcomes.

    For each plant, over {all, per-phase}, for signal in {level, change}, for
    every driver × outcome pair: Spearman rho + p, with a Benjamini–Hochberg
    q-value across all rows.

    对每株、各阶段、各信号（水平/变化），计算环境驱动与层位结果的 Spearman 相关，
    并做 Benjamini–Hochberg 校正。
    """
    from scipy import stats as _stats

    drivers = list(driver_columns)
    outcomes = list(outcome_columns)
    rows: List[Dict[str, Any]] = []
    for plant_id, frame in frames.items():
        subsets = {"all": frame}
        if phase_column is not None and phase_column in frame.columns:
            for phase, subset in frame.groupby(phase_column, sort=False):
                subsets[str(phase)] = subset
        for phase, subset in subsets.items():
            for signal in ("level", "change"):
                for driver in drivers:
                    for outcome in outcomes:
                        x_col = driver if signal == "level" else "d_%s" % driver
                        y_col = outcome if signal == "level" else "d_%s" % outcome
                        if x_col not in subset.columns or y_col not in subset.columns:
                            continue
                        # Align the two columns by index. Renaming avoids the
                        # duplicate-label trap when x_col == y_col (subset[[x,y]]
                        # then keeps two identically-named columns, so pair[x_col]
                        # selects a DataFrame and .nunique() raises). The
                        # x_col == y_col case is the degenerate self-pair:
                        # Spearman of a series with itself is rho=1.0, p=nan.
                        pair = pd.concat(
                            [subset[x_col].rename("__x__"),
                             subset[y_col].rename("__y__")],
                            axis=1,
                        ).dropna()
                        if (
                            len(pair) < 3
                            or pair["__x__"].nunique() < 2
                            or pair["__y__"].nunique() < 2
                        ):
                            rho = p = float("nan")
                        else:
                            result = _stats.spearmanr(pair["__x__"], pair["__y__"])
                            rho = float(result.statistic)
                            p = float(result.pvalue)
                        rows.append(
                            {
                                "plant_id": plant_id,
                                "phase": phase,
                                "signal": signal,
                                "driver": driver,
                                "outcome": outcome,
                                "n": int(len(pair)),
                                "spearman_rho": rho,
                                "p_value": p,
                            }
                        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_value_bh"] = _benjamini_hochberg(
            result["p_value"].to_numpy(dtype=float)
        )
    return result


def pair_shifted_times(
    df_shift: pd.DataFrame,
    df_target: pd.DataFrame,
    shift_hours: float = 24.0,
    tolerance_sec: float = 180.0,
    time_column: str = "timestamp",
    metric_columns: Optional[Sequence[str]] = None,
    phase_column: Optional[str] = None,
) -> pd.DataFrame:
    """Pair rows of ``df_shift`` (shifted by ``shift_hours``) to nearest ``df_target``.

    Uses ``pandas.merge_asof`` (nearest, within ``tolerance_sec``). Adds
    ``diff__<metric>`` columns (target − shift) for each ``metric_columns`` and,
    when ``phase_column`` is present in both frames, a ``phase_agreement`` flag.

    将 ``df_shift`` 的行平移 ``shift_hours`` 后，按最近时刻（容差 tolerance_sec）
    与 ``df_target`` 配对。对每个 metric 生成 ``diff__<metric>``（目标 − 平移），
    并在两帧均含 phase_column 时附上 phase_agreement 标志。
    """
    left = df_shift.copy()
    right = df_target.copy()
    left["_pair_target"] = pd.to_datetime(left[time_column]) + pd.Timedelta(
        hours=shift_hours
    )
    left = left.sort_values("_pair_target")
    right = right.sort_values(time_column)
    paired = pd.merge_asof(
        left,
        right,
        left_on="_pair_target",
        right_on=time_column,
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_sec),
        suffixes=("_shift", "_target"),
    )
    # After merge_asof the right time column is renamed "<time_column>_target";
    # rows with no in-tolerance match leave it NaN, so drop them there.
    # merge_asof 后右侧时间列被改名 "<time_column>_target"；容差内未配对的行该列为
    # NaN，据此剔除。
    target_time_col = "%s_target" % time_column
    paired = paired.dropna(subset=[target_time_col]).copy()
    if metric_columns is not None:
        for metric in metric_columns:
            s_col = "%s_shift" % metric
            t_col = "%s_target" % metric
            if s_col in paired.columns and t_col in paired.columns:
                paired["diff__%s" % metric] = paired[t_col] - paired[s_col]
    if (
        phase_column is not None
        and "%s_shift" % phase_column in paired.columns
        and "%s_target" % phase_column in paired.columns
    ):
        paired["phase_agreement"] = (
            paired["%s_target" % phase_column]
            == paired["%s_shift" % phase_column]
        )
    paired["pair_offset_sec"] = (
        paired[target_time_col] - paired["_pair_target"]
    ).dt.total_seconds()
    return paired
