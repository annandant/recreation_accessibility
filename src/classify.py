"""Классификация зон дефицита/нормы/профицита.

Важно: пороги Q25/Q75 фиксируются один раз на baseline-сценарии и
переиспользуются во всех последующих сценариях, иначе доля «дефицита»
по построению всегда оказывается равной 25 % (см. правки.txt).
"""
from __future__ import annotations
import geopandas as gpd
import pandas as pd


def classify_deficit_baseline(buildings: gpd.GeoDataFrame, cfg: dict) -> tuple[gpd.GeoDataFrame, float, float]:
    """Вычислить пороги по baseline и вернуть (buildings_with_class, q_low, q_high)."""
    q_low = float(buildings["A_h"].quantile(cfg["deficit"]["q_low"]))
    q_high = float(buildings["A_h"].quantile(cfg["deficit"]["q_high"]))

    return classify_with_thresholds(buildings, q_low, q_high), q_low, q_high


def classify_with_thresholds(buildings: gpd.GeoDataFrame, q_low: float, q_high: float) -> gpd.GeoDataFrame:
    """Применить фиксированные пороги."""
    bld = buildings.copy()
    def _cls(v):
        if v < q_low:
            return "deficit"
        if v > q_high:
            return "surplus"
        return "normal"
    bld["access_class"] = bld["A_h"].apply(_cls)
    bld["q_low_threshold"] = q_low
    bld["q_high_threshold"] = q_high
    return bld
