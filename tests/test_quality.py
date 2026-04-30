"""Юнит-тесты коэффициентов качества рекреационных объектов (таблица 6 НИР).

Проверяются:
    * граничные значения k_ndvi, k_water, k_road через вспомогательные
      функции _ndvi_to_coef, _water_dist_to_coef, _road_dist_to_coef;
    * функция k_zone на синтетическом GeoDataFrame зонирования;
    * calculate_quality_factor = 1 при всех k_* = 1;
    * calculate_quality_factor реагирует на отклонение от 1.
"""
from __future__ import annotations
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

from src.quality import (
    _ndvi_to_coef,
    _water_dist_to_coef,
    _road_dist_to_coef,
    calculate_quality_factor,
    k_zone,
)

# -------------------------------------------------------------------- #
# AHP-веса из config.yaml
# -------------------------------------------------------------------- #
_W = {"zone": 0.263, "ndvi": 0.558, "water": 0.088, "road": 0.091}


# -------------------------------------------------------------------- #
# k_ndvi — граничные значения (таблица 6 НИР)
# -------------------------------------------------------------------- #

def test_ndvi_coef_low():
    """NDVI < 0.33 → k=0.50."""
    assert _ndvi_to_coef(0.00) == 0.50
    assert _ndvi_to_coef(0.10) == 0.50
    assert _ndvi_to_coef(0.32) == 0.50


def test_ndvi_coef_medium():
    """0.33 ≤ NDVI ≤ 0.66 → k=1.00."""
    assert _ndvi_to_coef(0.33) == 1.00
    assert _ndvi_to_coef(0.50) == 1.00
    assert _ndvi_to_coef(0.66) == 1.00


def test_ndvi_coef_high():
    """NDVI > 0.66 → k=1.30."""
    assert _ndvi_to_coef(0.67) == 1.30
    assert _ndvi_to_coef(0.80) == 1.30
    assert _ndvi_to_coef(1.00) == 1.30


def test_ndvi_coef_boundary_33():
    """Граница 0.33 принадлежит medium (k=1.00), 0.3299… — low (k=0.50)."""
    assert _ndvi_to_coef(0.33) == 1.00
    assert _ndvi_to_coef(0.3299) == 0.50


def test_ndvi_coef_boundary_66():
    """Граница 0.66 принадлежит medium (k=1.00), 0.6601 — high (k=1.30)."""
    assert _ndvi_to_coef(0.66) == 1.00
    assert _ndvi_to_coef(0.6601) == 1.30


# -------------------------------------------------------------------- #
# k_water — граничные значения (таблица 6 НИР)
# -------------------------------------------------------------------- #

def test_water_coef_very_close():
    """d < 100 м → k=1.30."""
    assert _water_dist_to_coef(0.0)  == 1.30
    assert _water_dist_to_coef(50.0) == 1.30
    assert _water_dist_to_coef(99.9) == 1.30


def test_water_coef_close():
    """100 ≤ d ≤ 300 → k=1.20."""
    assert _water_dist_to_coef(100.0) == 1.20
    assert _water_dist_to_coef(200.0) == 1.20
    assert _water_dist_to_coef(300.0) == 1.20


def test_water_coef_medium():
    """300 < d ≤ 800 → k=1.10."""
    assert _water_dist_to_coef(301.0) == 1.10
    assert _water_dist_to_coef(600.0) == 1.10
    assert _water_dist_to_coef(800.0) == 1.10


def test_water_coef_far():
    """d > 800 → k=1.00."""
    assert _water_dist_to_coef(801.0) == 1.00
    assert _water_dist_to_coef(2000.0) == 1.00


# -------------------------------------------------------------------- #
# k_road — граничные значения (таблица 6 НИР)
# -------------------------------------------------------------------- #

def test_road_coef_very_close():
    """d ≤ 50 → k=0.50."""
    assert _road_dist_to_coef(0.0)  == 0.50
    assert _road_dist_to_coef(50.0) == 0.50


def test_road_coef_close():
    """50 < d ≤ 150 → k=0.65."""
    assert _road_dist_to_coef(51.0)  == 0.65
    assert _road_dist_to_coef(150.0) == 0.65


def test_road_coef_medium():
    """150 < d ≤ 300 → k=0.80."""
    assert _road_dist_to_coef(151.0) == 0.80
    assert _road_dist_to_coef(300.0) == 0.80


def test_road_coef_moderate():
    """300 < d ≤ 500 → k=0.90."""
    assert _road_dist_to_coef(301.0) == 0.90
    assert _road_dist_to_coef(500.0) == 0.90


def test_road_coef_far():
    """d > 500 → k=1.00."""
    assert _road_dist_to_coef(501.0) == 1.00
    assert _road_dist_to_coef(1000.0) == 1.00


# -------------------------------------------------------------------- #
# calculate_quality_factor — все k=1 → factor=1
# -------------------------------------------------------------------- #

def _rec_with_k(k_z, k_n, k_w, k_r) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"k_zone": [k_z], "k_ndvi": [k_n], "k_water": [k_w], "k_road": [k_r]},
        geometry=[Point(0, 0)],
        crs="EPSG:32636",
    )


def test_quality_factor_all_ones():
    """quality_factor = 1 при k_zone = k_ndvi = k_water = k_road = 1."""
    rec = _rec_with_k(1.0, 1.0, 1.0, 1.0)
    factor = calculate_quality_factor(rec, _W)
    assert abs(float(factor.iloc[0]) - 1.0) < 1e-9, (
        f"Ожидался factor=1, получено {float(factor.iloc[0])}"
    )


def test_quality_factor_above_one_for_high_ndvi():
    """Высокий k_ndvi (1.30) → factor > 1."""
    rec = _rec_with_k(1.0, 1.30, 1.0, 1.0)
    factor = calculate_quality_factor(rec, _W)
    assert float(factor.iloc[0]) > 1.0, "Ожидался factor > 1 при k_ndvi=1.30"


def test_quality_factor_below_one_for_bad_road():
    """Близкая дорога k_road=0.50 → factor < 1."""
    rec = _rec_with_k(1.0, 1.0, 1.0, 0.50)
    factor = calculate_quality_factor(rec, _W)
    assert float(factor.iloc[0]) < 1.0, "Ожидался factor < 1 при k_road=0.50"


def test_quality_factor_formula():
    """Ручная проверка формулы 1 + Σ w_k*(k_j − 1)."""
    k_z, k_n, k_w, k_r = 1.20, 1.30, 1.20, 0.90
    expected = (
        1
        + _W["zone"]  * (k_z - 1)
        + _W["ndvi"]  * (k_n - 1)
        + _W["water"] * (k_w - 1)
        + _W["road"]  * (k_r - 1)
    )
    rec = _rec_with_k(k_z, k_n, k_w, k_r)
    result = float(calculate_quality_factor(rec, _W).iloc[0])
    assert abs(result - expected) < 1e-9, (
        f"Ожидалось {expected:.8f}, получено {result:.8f}"
    )


# -------------------------------------------------------------------- #
# k_zone — синтетическое зонирование
# -------------------------------------------------------------------- #

def _make_zone_gdf(label: str, cx: float = 0.0, cy: float = 0.0, r: float = 500.0):
    """Полигон зонирования с атрибутом 'zone'."""
    poly = Point(cx, cy).buffer(r)
    return gpd.GeoDataFrame(
        {"zone": [label]},
        geometry=[poly],
        crs="EPSG:32636",
    )


def _make_rec_at(cx: float, cy: float) -> gpd.GeoDataFrame:
    """Точечный объект рекреации с центроидом."""
    pt = Point(cx, cy)
    rec = gpd.GeoDataFrame(geometry=[pt], crs="EPSG:32636")
    rec["centroid"] = rec.geometry
    # заменяем геометрию на точечную (k_zone использует centroid)
    return rec


def test_k_zone_recreational():
    """Центроид в рекреационной зоне (Р1) → k_zone = 1.20."""
    zoning = _make_zone_gdf("Р1")
    rec = _make_rec_at(0.0, 0.0)
    result = k_zone(rec, zoning)
    assert abs(float(result.iloc[0]) - 1.20) < 1e-9, (
        f"Ожидался k_zone=1.20, получено {float(result.iloc[0])}"
    )


def test_k_zone_industrial():
    """Центроид в производственной зоне (П) → k_zone = 0.60."""
    zoning = _make_zone_gdf("П")
    rec = _make_rec_at(0.0, 0.0)
    result = k_zone(rec, zoning)
    assert abs(float(result.iloc[0]) - 0.60) < 1e-9, (
        f"Ожидался k_zone=0.60, получено {float(result.iloc[0])}"
    )


def test_k_zone_null_is_water():
    """Null в label (акватория) → k_zone = 1.15."""
    import numpy as np
    zoning = _make_zone_gdf(None)    # NaN → акватория
    rec = _make_rec_at(0.0, 0.0)
    result = k_zone(rec, zoning)
    assert abs(float(result.iloc[0]) - 1.15) < 1e-9, (
        f"Ожидался k_zone=1.15 для null-зоны, получено {float(result.iloc[0])}"
    )


def test_k_zone_forbidden():
    """Запретная зона (СП1) → k_zone = 0.10."""
    zoning = _make_zone_gdf("СП1")
    rec = _make_rec_at(0.0, 0.0)
    result = k_zone(rec, zoning)
    assert abs(float(result.iloc[0]) - 0.10) < 1e-9, (
        f"Ожидался k_zone=0.10 для запретной зоны, получено {float(result.iloc[0])}"
    )


def test_k_zone_none_zoning():
    """При zoning=None → k_zone = 1.00 для всех объектов."""
    rec = _make_rec_at(0.0, 0.0)
    result = k_zone(rec, None)
    assert float(result.iloc[0]) == 1.00


# -------------------------------------------------------------------- #

if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  OK  · {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL · {t.__name__}: {e}")
    if failures == 0:
        print(f"\nВсе {len(tests)} тестов пройдены.")
    else:
        print(f"\nНЕ ПРОЙДЕНО: {failures} из {len(tests)}")
        sys.exit(1)
