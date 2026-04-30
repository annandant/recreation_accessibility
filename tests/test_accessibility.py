"""Юнит-тесты алгоритма Huff-2SFCA (формулы 3–10 НИР).

Используется полностью синтетический граф расстояний (dist_matrix),
без OSM и без реальных файлов.

Тестируемые свойства
--------------------
1. Положительность A_h при наличии доступных объектов.
2. A_h = 0, если объектов в радиусе нет.
3. Симметрия: зеркальные здания → одинаковый A_h.
4. Монотонность: ближе → выше A_h.
5. Единицы: A_h — безразмерная величина (отношение вместимостей),
   обязательно в [0, +inf); при умеренной нагрузке ≤ 1.

Расположение синтетических данных
----------------------------------
Два парка P0, P1 (local, capacity_adj=50 чел).
Три здания B0, B1, B2 (pop=100 каждый):
    B0  ←  d=200  →  P0   (P1 недоступен — расстояние > 400 м)
    B1  ←  d=200  →  P1   (P0 недоступен — расстояние > 400 м)
    B2  ←  d=300  →  P0 и P1  (оба в радиусе 400 м)

При весах [local=1.0, district=0.0, city=0.0] A_h = A_local.

Аналитический расчёт (beta=0.004, lambda=1):
    H_ij[B0] = {P0: 1.0},  H_ij[B1] = {P1: 1.0}
    H_ij[B2] = {P0: 0.5, P1: 0.5}
    P_j[P0] = 100×1.0 + 100×0.5 = 150,  P_j[P1] = 150
    R_j[P0] = R_j[P1] = 50/150 ≈ 0.3333
    A_h[B0] = A_h[B1] = (1/3)×exp(-0.4) ≈ 0.1498  ← симметрия
    A_h[B2] = (2/3)×exp(-0.8) ≈ 0.2996
"""
from __future__ import annotations
import math
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

from src.accessibility import calculate_accessibility

# -------------------------------------------------------------------- #
# Общие константы теста
# -------------------------------------------------------------------- #
_CFG = {
    "service_levels": {
        "local":    {"radius": 400,  "area_max": 30_000,    "beta": 0.004,  "weight": 1.0},
        "district": {"radius": 1500, "area_max": 150_000,   "beta": 0.0015, "weight": 0.0},
        "city":     {"radius": 6000, "area_max": 1e12,      "beta": 0.0004, "weight": 0.0},
    },
    "huff": {"lambda": 1.0},
}

_BETA = _CFG["service_levels"]["local"]["beta"]


def _make_square(cx: float, cy: float, side: float = 100.0) -> Polygon:
    """Квадратный полигон со стороной `side` вокруг центра (cx, cy)."""
    h = side / 2
    return Polygon([
        (cx - h, cy - h), (cx + h, cy - h),
        (cx + h, cy + h), (cx - h, cy + h),
    ])


def _make_buildings(coords: list[tuple[float, float]], pops: list[int]) -> gpd.GeoDataFrame:
    geoms = [Point(x, y) for x, y in coords]
    return gpd.GeoDataFrame(
        {"population": pops},
        geometry=geoms,
        crs="EPSG:32636",
    )


def _make_recreation(
    coords: list[tuple[float, float]],
    cap_adj: float = 50.0,
    level: str = "local",
    side: float = 100.0,
) -> gpd.GeoDataFrame:
    geoms = [_make_square(x, y, side) for x, y in coords]
    rec = gpd.GeoDataFrame(
        {
            "capacity":     [cap_adj] * len(coords),
            "capacity_adj": [cap_adj] * len(coords),
            "service_level": [level]  * len(coords),
        },
        geometry=geoms,
        crs="EPSG:32636",
    )
    rec["centroid"] = rec.geometry.centroid
    return rec


def _symmetric_setup():
    """Базовая симметричная конфигурация 3 здания × 2 парка."""
    buildings = _make_buildings(
        [(0, 0), (1000, 0), (500, 0)],
        [100, 100, 100],
    )
    recreation = _make_recreation([(0, 0), (1000, 0)])
    dist_matrix: dict[int, dict[int, float]] = {
        0: {0: 200},         # B0 → P0, P1 > 400 м
        1: {1: 200},         # B1 → P1, P0 > 400 м
        2: {0: 300, 1: 300}, # B2 → оба в радиусе
    }
    return buildings, recreation, dist_matrix


# -------------------------------------------------------------------- #
# Тест 1: положительность A_h
# -------------------------------------------------------------------- #
def test_ah_positive_when_accessible():
    """A_h > 0 для всех зданий, имеющих хотя бы один доступный парк."""
    bld, rec, dm = _symmetric_setup()
    bld_out, _, _ = calculate_accessibility(bld, rec, dm, _CFG)
    assert (bld_out["A_h"] > 0).all(), (
        f"Ожидались A_h > 0 для всех, получено: {bld_out['A_h'].tolist()}"
    )


# -------------------------------------------------------------------- #
# Тест 2: нулевой A_h при отсутствии парков в радиусе
# -------------------------------------------------------------------- #
def test_ah_zero_when_no_park_in_radius():
    """Здание без парков в радиусе → A_h = 0."""
    bld = _make_buildings([(5000, 5000)], [100])  # далеко от всего
    rec = _make_recreation([(0, 0)])
    dm: dict[int, dict[int, float]] = {0: {}}  # нет доступных парков
    bld_out, _, _ = calculate_accessibility(bld, rec, dm, _CFG)
    assert bld_out["A_h"].iloc[0] == 0.0, (
        f"Ожидался A_h=0, получено: {bld_out['A_h'].iloc[0]}"
    )


# -------------------------------------------------------------------- #
# Тест 3: симметрия
# -------------------------------------------------------------------- #
def test_symmetry_equal_ah():
    """Зеркальные здания (B0 ↔ B1) → одинаковый A_h."""
    bld, rec, dm = _symmetric_setup()
    bld_out, _, _ = calculate_accessibility(bld, rec, dm, _CFG)
    A0 = bld_out["A_h"].iloc[0]
    A1 = bld_out["A_h"].iloc[1]
    assert abs(A0 - A1) < 1e-9, (
        f"Симметрия нарушена: A_h[B0]={A0:.6f}, A_h[B1]={A1:.6f}"
    )


# -------------------------------------------------------------------- #
# Тест 4: аналитическое значение (верификация формул)
# -------------------------------------------------------------------- #
def test_ah_analytical_value():
    """A_h[B0] сравнивается с аналитически вычисленным значением."""
    bld, rec, dm = _symmetric_setup()
    bld_out, _, _ = calculate_accessibility(bld, rec, dm, _CFG)

    # Аналитика (см. docstring модуля):
    # R_j = 50 / 150 = 1/3
    # A_h[B0] = R_j[P0] * exp(-beta * 200)
    R_j_expected = 50.0 / 150.0
    A_expected = R_j_expected * math.exp(-_BETA * 200)

    A0 = float(bld_out["A_h"].iloc[0])
    assert abs(A0 - A_expected) < 1e-9, (
        f"A_h[B0]: ожидалось {A_expected:.8f}, получено {A0:.8f}"
    )


# -------------------------------------------------------------------- #
# Тест 5: монотонность (ближе → выше A_h при прочих равных)
# -------------------------------------------------------------------- #
def test_monotonicity_closer_higher_ah():
    """B_near (d=100) лучше обеспечен, чем B_far (d=300)."""
    buildings = _make_buildings([(0, 0), (1000, 0)], [100, 100])
    recreation = _make_recreation([(0, 0)])
    dm: dict[int, dict[int, float]] = {
        0: {0: 100},  # B_near
        1: {0: 300},  # B_far
    }
    bld_out, _, _ = calculate_accessibility(buildings, recreation, dm, _CFG)
    assert bld_out["A_h"].iloc[0] > bld_out["A_h"].iloc[1], (
        "Ближнее здание должно иметь бо́льший A_h"
    )


# -------------------------------------------------------------------- #
# Тест 6: единицы и диапазон значений
# -------------------------------------------------------------------- #
def test_ah_dimensionless_and_bounded():
    """A_h — безразмерный показатель ≥ 0; при типичной нагрузке ≤ 5."""
    bld, rec, dm = _symmetric_setup()
    bld_out, _, metrics = calculate_accessibility(bld, rec, dm, _CFG)
    assert (bld_out["A_h"] >= 0).all(), "A_h не может быть отрицательным"
    # при capacity_adj=50 и population=100 разумный максимум << 5
    assert bld_out["A_h"].max() < 5.0, (
        f"Подозрительно высокий A_h: {bld_out['A_h'].max()}"
    )


# -------------------------------------------------------------------- #
# Тест 7: метрика A_m = суммарная вместимость / суммарное население
# -------------------------------------------------------------------- #
def test_A_m_formula():
    """A_m = Σ capacity_adj / Σ population."""
    bld, rec, dm = _symmetric_setup()
    _, _, metrics = calculate_accessibility(bld, rec, dm, _CFG)
    expected_A_m = (50.0 + 50.0) / (100.0 + 100.0 + 100.0)  # 100/300 ≈ 0.333
    assert abs(metrics["A_m"] - expected_A_m) < 1e-9, (
        f"A_m: ожидалось {expected_A_m:.6f}, получено {metrics['A_m']:.6f}"
    )


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
