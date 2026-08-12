# -*- coding: utf-8 -*-
"""Environment-sensor time alignment for thermal frames (no extrapolation).

Bridges an environment time-series (air temperature / CO₂ / soil moisture /
light ...) onto each FLIR frame timestamp by **linear interpolation between the
two bracketing observations**. Two hard rules make this fail-closed:

  * **No extrapolation** — a frame whose timestamp falls outside the sensor
    range returns ``NaN`` + ``qc_flag = "outside_sensor_range"``.
  * **Gap guard** — when the bracketing observations are farther apart than
    ``max_gap_sec``, the value is ``NaN`` + ``qc_flag = "sensor_gap_exceeds_limit"``
    (missing data would otherwise be silently invented).

Ported (desensitised, bilingual) from the private FLIR pipeline:
``parse_flir_stem``, ``prepare_sensor_frame``, ``interpolate_sensor_frame``,
``InterpolationResult``, plus ``read_sensor_workbook`` / ``read_environment_workbook``
and the :class:`EnvironmentJoiner` convenience wrapper.

环境传感器时序对齐到热红外帧（禁止外推）。

用目标帧时刻前后两个观测做线性插值，把环境时序（空气温度/CO₂/土壤湿度/光照…）
对齐到每个 FLIR 帧。两条 fail-closed 硬规则：禁止外推（超出传感器范围返回 NaN +
qc_flag）、缺口超限（观测间隔超过 max_gap_sec 返回 NaN + qc_flag）。全部脱敏并保留
中英双语说明。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Interpolation primitives / 插值原语
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class InterpolationResult:
    """Single-target-time interpolation result. 单目标时点的插值结果。"""

    values: Dict[str, float]
    previous_time: Optional[pd.Timestamp]
    next_time: Optional[pd.Timestamp]
    bracket_gap_sec: Optional[float]
    nearest_offset_sec: Optional[float]
    qc_flag: str


def _empty_result(flag: str) -> InterpolationResult:
    return InterpolationResult(
        values={},
        previous_time=None,
        next_time=None,
        bracket_gap_sec=None,
        nearest_offset_sec=None,
        qc_flag=flag,
    )


def parse_flir_stem(stem: str, timezone: str = "UTC") -> datetime:
    """Parse a ``YYYYMMDD_HHMMSS_mmm`` local capture instant.

    解析 ``YYYYMMDD_HHMMSS_mmm`` 格式的本地采集时刻（带时区）。
    """
    naive = datetime.strptime(stem, "%Y%m%d_%H%M%S_%f")
    return naive.replace(tzinfo=ZoneInfo(timezone))


def prepare_sensor_frame(
    frame: pd.DataFrame,
    timestamp_column: str,
    timezone: str = "UTC",
) -> tuple[pd.DataFrame, set]:
    """Normalise, sort and de-duplicate sensor timestamps; return dup set for QC.

    Timestamps are localised/converted to ``timezone`` and sorted. Duplicate
    timestamps are aggregated to their mean (numeric columns) so that a
    deterministic, gap-aware interpolation is possible. Returns the cleaned
    frame and the set of duplicated timestamps (for downstream QC flags).

    规范化、排序并去重传感器时间戳：转换到指定时区、排序；重复的timestamp取均值
    （数值列）以得到确定性的间隙感知插值。返回清洗后的帧与重复时间戳集合（供 QC）。
    """
    prepared = frame.copy()
    parsed = pd.to_datetime(prepared[timestamp_column], errors="coerce")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(timezone)
    else:
        parsed = parsed.dt.tz_convert(timezone)
    prepared[timestamp_column] = parsed
    prepared = prepared.dropna(subset=[timestamp_column]).sort_values(
        timestamp_column
    )

    duplicate_mask = prepared.duplicated(timestamp_column, keep=False)
    duplicate_times = set(
        prepared.loc[duplicate_mask, timestamp_column].tolist()
    )
    if duplicate_times:
        numeric_columns = [
            column
            for column in prepared.columns
            if column != timestamp_column
            and pd.api.types.is_numeric_dtype(prepared[column])
        ]
        aggregations = {column: "mean" for column in numeric_columns}
        prepared = (
            prepared[[timestamp_column, *numeric_columns]]
            .groupby(timestamp_column, as_index=False)
            .agg(aggregations)
        )
    return prepared.reset_index(drop=True), duplicate_times


def interpolate_sensor_frame(
    frame: pd.DataFrame,
    target_time: Union[datetime, pd.Timestamp],
    timestamp_column: str,
    value_columns: List[str],
    max_gap_sec: float = 600.0,
    duplicate_times: Optional[set] = None,
) -> InterpolationResult:
    """Linearly interpolate sensor values at ``target_time``; never extrapolate.

    用目标时点前后的观测做线性插值；禁止外推、拒绝跨越过大缺口。

    Returns an :class:`InterpolationResult` with ``NaN`` values + a non-``ok``
    ``qc_flag`` whenever: the frame is empty, the target is outside the sensor
    range, the bracket gap exceeds ``max_gap_sec``, a bracket value is missing,
    or the bracket gap is non-positive.
    """
    if frame.empty:
        return _empty_result("sensor_empty")

    target = pd.Timestamp(target_time)
    times = frame[timestamp_column]

    # Reconcile timezone awareness so searchsorted / comparisons never raise on
    # a tz-naive target against a tz-aware sensor series (and vice versa).
    # 统一时区感知，避免 tz-naive 目标与 tz-aware 传感器序列比较时报错。
    ttz = times.dt.tz
    if ttz is not None and target.tzinfo is None:
        target = target.tz_localize(ttz)
    elif ttz is None and target.tzinfo is not None:
        times = times.dt.tz_localize(target.tzinfo)
    elif ttz is not None and target.tzinfo is not None and ttz != target.tzinfo:
        target = target.tz_convert(ttz)

    position = int(times.searchsorted(target, side="left"))
    if position < len(frame) and pd.Timestamp(times.iloc[position]) == target:
        exact = frame.iloc[position]
        values = {}
        for column in value_columns:
            raw = pd.to_numeric(pd.Series([exact[column]]), errors="coerce").iloc[0]
            values[column] = float(raw) if pd.notna(raw) else np.nan
        duplicate_times = duplicate_times or set()
        flag = (
            "duplicate_sensor_timestamp_aggregated"
            if target in duplicate_times
            else "ok"
        )
        if any(not np.isfinite(value) for value in values.values()):
            flag = "sensor_value_missing"
        return InterpolationResult(
            values=values,
            previous_time=target,
            next_time=target,
            bracket_gap_sec=0.0,
            nearest_offset_sec=0.0,
            qc_flag=flag,
        )
    if position == 0 or position >= len(frame):
        return _empty_result("outside_sensor_range")

    previous = frame.iloc[position - 1]
    following = frame.iloc[position]
    previous_time = pd.Timestamp(previous[timestamp_column])
    next_time = pd.Timestamp(following[timestamp_column])
    gap_sec = float((next_time - previous_time).total_seconds())
    nearest_offset = min(
        abs(float((target - previous_time).total_seconds())),
        abs(float((next_time - target).total_seconds())),
    )

    if gap_sec <= 0:
        return InterpolationResult(
            values={column: np.nan for column in value_columns},
            previous_time=previous_time,
            next_time=next_time,
            bracket_gap_sec=gap_sec,
            nearest_offset_sec=nearest_offset,
            qc_flag="nonpositive_time_gap",
        )
    if gap_sec > max_gap_sec:
        return InterpolationResult(
            values={column: np.nan for column in value_columns},
            previous_time=previous_time,
            next_time=next_time,
            bracket_gap_sec=gap_sec,
            nearest_offset_sec=nearest_offset,
            qc_flag="sensor_gap_exceeds_limit",
        )

    fraction = float((target - previous_time).total_seconds()) / gap_sec
    values: Dict[str, float] = {}
    for column in value_columns:
        before = pd.to_numeric(pd.Series([previous[column]]), errors="coerce").iloc[0]
        after = pd.to_numeric(pd.Series([following[column]]), errors="coerce").iloc[0]
        if pd.isna(before) or pd.isna(after):
            values[column] = np.nan
        else:
            values[column] = float(before + fraction * (after - before))

    duplicate_times = duplicate_times or set()
    qc_flag = (
        "duplicate_sensor_timestamp_aggregated"
        if previous_time in duplicate_times or next_time in duplicate_times
        else "ok"
    )
    if any(not np.isfinite(value) for value in values.values()):
        qc_flag = "sensor_value_missing"

    return InterpolationResult(
        values=values,
        previous_time=previous_time,
        next_time=next_time,
        bracket_gap_sec=gap_sec,
        nearest_offset_sec=nearest_offset,
        qc_flag=qc_flag,
    )


# --------------------------------------------------------------------------
# Workbook readers / 工作簿读取
# --------------------------------------------------------------------------
def read_sensor_workbook(
    path: Union[str, Path],
    column_map: Dict[str, str],
    sheet_name: Union[str, int] = 0,
) -> pd.DataFrame:
    """Read a sensor workbook and unify semantic column names.

    ``column_map`` maps raw source columns -> semantic names (e.g. a raw
    "AirTemp" -> "ambient_c"). The ``timestamp`` column (if present) is kept
    as-is and never coerced to numeric. Raises ``ValueError`` when a required
    source column is absent.

    读取传感器工作簿并统一语义字段名。``column_map`` 将原始列映射到语义名（如
    "AirTemp" → "ambient_c"）。``timestamp`` 列保留原样、不做数值化。缺失必需
    源列时抛出 ValueError。
    """
    frame = pd.read_excel(path, sheet_name=sheet_name)
    missing = [source for source in column_map if source not in frame.columns]
    if missing:
        raise ValueError("Sensor workbook missing columns: %s" % missing)
    selected = frame[list(column_map)].rename(columns=column_map)
    for column in selected.columns:
        if column != "timestamp":
            selected[column] = pd.to_numeric(selected[column], errors="coerce")
    return selected


def read_environment_workbook(
    path: Union[str, Path],
    column_map: Dict[str, str],
    sheet_name: Union[str, int] = 0,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Read an environment workbook via :func:`read_sensor_workbook`.

    A thin wrapper that also asserts the presence of ``timestamp_column`` so
    the resulting frame can be consumed by :class:`EnvironmentJoiner` /
    :func:`align_environment_to_frames` without further checks.

    通过 :func:`read_sensor_workbook` 读取环境工作簿的薄包装，并校验时间戳列存在，
    以便直接交给 :class:`EnvironmentJoiner` / :func:`align_environment_to_frames`。
    """
    df = read_sensor_workbook(path, column_map, sheet_name=sheet_name)
    if timestamp_column not in df.columns:
        raise ValueError(
            "Environment workbook must contain a '%s' column." % timestamp_column
        )
    return df


# --------------------------------------------------------------------------
# Joiner / 对齐器
# --------------------------------------------------------------------------
class EnvironmentJoiner:
    """Stateful wrapper that aligns an environment frame onto many target times.

    有状态包装器：把一个环境数据帧多次对齐到不同目标时刻。"""

    def __init__(
        self,
        env_dataframe: pd.DataFrame,
        value_columns: List[str],
        timestamp_column: str = "timestamp",
        timezone: str = "UTC",
        max_gap_sec: float = 600.0,
    ):
        self.frame, self.duplicate_times = prepare_sensor_frame(
            env_dataframe, timestamp_column, timezone
        )
        self.value_columns = list(value_columns)
        self.timestamp_column = timestamp_column
        self.max_gap_sec = max_gap_sec

    def align(self, target_time: Union[datetime, pd.Timestamp]) -> InterpolationResult:
        """Align the environment to a single target time. 对齐到单个目标时刻。"""
        return interpolate_sensor_frame(
            self.frame,
            target_time,
            self.timestamp_column,
            self.value_columns,
            self.max_gap_sec,
            self.duplicate_times,
        )

    def align_many(self, target_times) -> pd.DataFrame:
        """Align the environment to many target times; return a tidy DataFrame.

        对齐到多个目标时刻，返回规整 DataFrame（含 qc_flag / bracket_gap_sec /
        nearest_offset_sec）。
        """
        rows = []
        for t in target_times:
            result = self.align(t)
            row = {"target_time": pd.Timestamp(t)}
            # Always emit every value column (NaN-filled) so that
            # empty / outside-range / gap results still produce a well-formed
            # DataFrame instead of raising KeyError downstream.
            # 始终输出每个值列（缺失则填 NaN），保证空/越界/断点结果也形成规整
            # DataFrame，避免下游 KeyError。
            for vc in self.value_columns:
                row[vc] = result.values.get(vc, float("nan"))
            row["qc_flag"] = result.qc_flag
            row["bracket_gap_sec"] = result.bracket_gap_sec
            row["nearest_offset_sec"] = result.nearest_offset_sec
            rows.append(row)
        return pd.DataFrame(rows)


def align_environment_to_frames(
    frame_timestamps,
    env_dataframe: pd.DataFrame,
    value_columns: List[str],
    max_gap_sec: float = 600.0,
    timestamp_column: str = "timestamp",
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Align an environment time-series onto a list of frame timestamps.

    For each frame timestamp, linearly interpolate the requested
    ``value_columns`` from ``env_dataframe``. Frames outside the sensor range
    or across a gap larger than ``max_gap_sec`` get ``NaN`` + a non-``ok``
    ``qc_flag`` (no extrapolation, no fabricated data).

    把环境时序按各帧时间戳线性插值对齐。超出传感器范围或跨越超过 max_gap_sec 的
    帧得到 NaN + 非 ok 的 qc_flag（禁止外推、不编造数据）。

    Returns
    -------
    pd.DataFrame
        One row per frame timestamp with the interpolated ``value_columns`` plus
        ``qc_flag``, ``bracket_gap_sec`` and ``nearest_offset_sec``.
    """
    joiner = EnvironmentJoiner(
        env_dataframe,
        value_columns,
        timestamp_column=timestamp_column,
        timezone=timezone,
        max_gap_sec=max_gap_sec,
    )
    return joiner.align_many(frame_timestamps)
