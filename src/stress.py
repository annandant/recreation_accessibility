"""Индекс рекреационного стресса.

Историческая формула (НИР, формула 11):
    S_A = 1 − (A_max − A_mean) / (A_max − A_min)

Проблема: значение зависит только от взаимного расположения min/mean/max
в данной выборке. При почти равных A_h (например, 0.18, 0.19, 0.20)
индекс ≈ 0.5 — независимо от того, насколько эта обеспеченность далека
от целевого норматива A_m. Поэтому на практике он перестаёт давать
репрезентативные значения и не сравним между территориями.

Заменяющая формула (используется здесь):
    S_i = max(0, 1 − A_h_i / A_m)                — на уровне дома
    S_A = Σ_i (P_i · S_i) / Σ_i P_i               — агрегат по территории

где A_m = Σ C'_j / Σ P_j — теоретический максимум при равномерном
распределении (формула 9 НИР). S_i ∈ [0, 1]: 0 — обеспеченность ≥ A_m,
1 — нулевая обеспеченность. Подробное обоснование см. МЕТОДИКА_СТРЕСС.md.
"""
from __future__ import annotations
import geopandas as gpd
import pandas as pd

EPS = 1e-9


def calculate_stress(
    buildings: gpd.GeoDataFrame,
    A_m: float,
) -> tuple[gpd.GeoDataFrame, float]:
    """Записать S_i на дома и вернуть (buildings, S_A).

    S_i  — индивидуальный стресс жилого дома, 0..1.
    S_A  — взвешенный по населению стресс по всей территории.
    """
    if A_m <= 0:
        raise ValueError(f"A_m должен быть > 0, получено {A_m}")
    bld = buildings.copy()
    s_i = (1.0 - bld["A_h"] / A_m).clip(lower=0.0, upper=1.0)
    bld["stress_i"] = s_i
    pop = bld["population"].astype(float)
    total_pop = float(pop.sum())
    if total_pop <= 0:
        return bld, 0.0
    S_A = float((pop * s_i).sum() / total_pop)
    return bld, S_A
