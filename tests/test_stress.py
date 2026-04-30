"""Юнит-тест нового индекса рекреационного стресса.

Запуск: python -m pytest tests/test_stress.py -v
        или просто: python tests/test_stress.py
"""
from __future__ import annotations
import sys
import pathlib

# чтобы тест запускался без установки пакета
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import geopandas as gpd
from shapely.geometry import Point

from src.stress import calculate_stress


def _toy(populations, A_h_values):
    geoms = [Point(i, 0) for i in range(len(populations))]
    bld = gpd.GeoDataFrame(
        {"population": populations, "A_h": A_h_values},
        geometry=geoms, crs="EPSG:32636",
    )
    return bld


# 1. Полная обеспеченность ⇒ S_T = 0.
def test_full_supply_zero_stress():
    bld = _toy([100, 200, 300], [1.0, 1.0, 1.0])
    _, S_T = calculate_stress(bld, A_m=1.0)
    assert abs(S_T - 0.0) < 1e-9


# 2. Нулевая обеспеченность ⇒ S_T = 1.
def test_no_supply_full_stress():
    bld = _toy([100, 200, 300], [0.0, 0.0, 0.0])
    _, S_T = calculate_stress(bld, A_m=1.0)
    assert abs(S_T - 1.0) < 1e-9


# 3. Монотонность по A_h.
def test_monotonic_in_A_h():
    base = _toy([100]*4, [0.10, 0.20, 0.30, 0.40])
    better = _toy([100]*4, [0.20, 0.30, 0.40, 0.50])
    _, S_base = calculate_stress(base, A_m=1.0)
    _, S_better = calculate_stress(better, A_m=1.0)
    assert S_better < S_base


# 4. Реакция на смещение населения в менее обеспеченные дома.
#    Замечание: при ПРОПОРЦИОНАЛЬНОМ росте населения (когда A_h и A_m
#    масштабируются одинаково) индекс инвариантен — это сознательное
#    свойство популяционно-взвешенной метрики. Реальный эффект сценария
#    демографического развития проявляется через РАСПРЕДЕЛЕНИЕ нового
#    населения: новые жители заселяются в недообеспеченные районы.
def test_population_shift_increases_stress():
    A_h = [0.2, 0.5, 0.8]
    bld_before = _toy([100, 100, 100], A_h)
    bld_after  = _toy([300, 100, 100], A_h)   # +200 чел в худший дом
    _, S_before = calculate_stress(bld_before, A_m=1.0)
    _, S_after = calculate_stress(bld_after, A_m=1.0)
    assert S_after > S_before, f"{S_after} should be > {S_before}"


# 5. Инвариантность к перестановке зданий (порядок строк не должен влиять).
def test_invariant_to_permutation():
    bld_a = _toy([100, 200, 300], [0.1, 0.2, 0.3])
    bld_b = _toy([300, 100, 200], [0.3, 0.1, 0.2])
    _, S_a = calculate_stress(bld_a, A_m=1.0)
    _, S_b = calculate_stress(bld_b, A_m=1.0)
    assert abs(S_a - S_b) < 1e-9


# 6. Один «суперобеспеченный» дом не должен обнулять стресс остальных.
def test_robust_to_outlier():
    # старая формула здесь дала бы S_A ≈ 0.05 (mean ≪ max),
    # новая — должна остаться высокой, т.к. большинство недообеспечено.
    bld = _toy([100, 100, 100, 1], [0.05, 0.05, 0.05, 5.0])
    _, S_T = calculate_stress(bld, A_m=1.0)
    assert S_T > 0.9, f"индекс не должен сжиматься выбросом, получено {S_T}"


# 7. Старая формула на этом же кейсе была бы нечувствительна.
def test_compares_with_naive_old_formula():
    bld = _toy([100, 100, 100, 1], [0.05, 0.05, 0.05, 5.0])
    A_h = bld["A_h"]
    S_old = 1 - (A_h.max() - A_h.mean()) / (A_h.max() - A_h.min())
    _, S_new = calculate_stress(bld, A_m=1.0)
    # новая формула репрезентативнее — стресс сильно выше
    assert S_new > S_old


if __name__ == "__main__":
    failures = 0
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
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
