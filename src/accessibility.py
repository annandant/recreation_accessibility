"""Расчёт обеспеченности

Этап 1 (Huff-2SFCA):
    f_d_ij = exp(−β · d_ij),
    H_ij   = (C'_j · f) / Σ_k (C'_k · f)        — выбор объекта,
    P_j    = Σ_i  H_ij · pop_i,                — спрос в зоне обслуживания,
    R_j    = C'_j / max(P_j, ε).

Этап 2:
    A_i(r) = Σ_j  R_j · f_d_ij,
    A_h    = w_l · A_400 + w_d · A_1500 + w_c · A_6000,
    A_m    = Σ C'_j / Σ P_j,
    CV     = σ(A_h) / mean(A_h).
"""
from __future__ import annotations
import math
from typing import Mapping

import numpy as np
import pandas as pd
import geopandas as gpd

EPS = 1e-9


def _candidates(
    dist_matrix: dict[int, dict[int, float]],
    rec_idx_pool: set[int],
    radius: float,
) -> dict[int, dict[int, float]]:
    """Вернуть подматрицу: только объекты в нужном радиусе и из заданного пула."""
    out: dict[int, dict[int, float]] = {}
    for b, dd in dist_matrix.items():
        out[b] = {r: d for r, d in dd.items() if r in rec_idx_pool and d <= radius}
    return out


def _compute_level(
    buildings: gpd.GeoDataFrame,
    recreation: gpd.GeoDataFrame,
    dist_matrix: dict[int, dict[int, float]],
    level: str,
    cfg: dict,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Расчёт A_i(r), R_j, P_j для одного уровня обслуживания.

    Оптимизация: значения A_i собираются в словарь и конвертируются
    в Series одним вызовом — это быстрее, чем .loc[b] внутри цикла.
    """
    sl = cfg["service_levels"][level]
    radius: float = sl["radius"]
    beta: float = sl["beta"]
    lam: float = cfg["huff"]["lambda"]

    rec_pool = set(recreation.index[recreation["service_level"] == level])
    cap: dict[int, float] = recreation["capacity_adj"].to_dict()

    sub = _candidates(dist_matrix, rec_pool, radius)

    # --- этап 1: H_ij ---------------------------------------------------
    H_ij: dict[int, dict[int, float]] = {}
    for b, dd in sub.items():
        if not dd:
            continue
        attractions = {r: cap.get(r, 0.0) * math.exp(-beta * d) for r, d in dd.items()}
        total = sum(a ** lam for a in attractions.values())
        if total <= 0:
            continue
        H_ij[b] = {r: (a ** lam) / total for r, a in attractions.items()}

    # --- этап 1: P_j (спрос на каждый объект) ---------------------------
    pop: dict[int, float] = buildings["population"].to_dict()
    P_j: dict[int, float] = {r: 0.0 for r in rec_pool}
    for b, dd in H_ij.items():
        for r, h in dd.items():
            P_j[r] += h * pop.get(b, 0)

    # --- этап 1: R_j = C'_j / max(P_j, eps) ----------------------------
    R_j: dict[int, float] = {
        r: cap.get(r, 0.0) / max(P_j[r], EPS) for r in rec_pool
    }

    # --- этап 2: A_i(r) = Σ_j R_j · f_d_ij ----------------------------
    # Собираем значения в dict, чтобы избежать накладных расходов .loc
    # при обращении к pd.Series в цикле (оптимизация).
    a_vals: dict[int, float] = {}
    for b, dd in sub.items():
        if not dd:
            continue
        a_vals[b] = sum(R_j[r] * math.exp(-beta * d) for r, d in dd.items())

    A_i = pd.Series(0.0, index=buildings.index, name=f"A_{radius}")
    if a_vals:
        A_i.update(pd.Series(a_vals))

    return (
        A_i,
        pd.Series(R_j, name=f"R_{level}"),
        pd.Series(P_j, name=f"P_{level}"),
    )


def calculate_accessibility(
    buildings: gpd.GeoDataFrame,
    recreation: gpd.GeoDataFrame,
    dist_matrix: dict[int, dict[int, float]],
    cfg: dict,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict]:
    """Полный расчёт обеспеченности по трём уровням и итоговый A_h.

    Возвращает обновлённые buildings/recreation и словарь сводных метрик.
    """
    bld = buildings.copy()
    rec = recreation.copy()
    sl = cfg["service_levels"]

    # три радиуса
    A_local, R_local, P_local = _compute_level(bld, rec, dist_matrix, "local", cfg)
    A_dist,  R_dist,  P_dist  = _compute_level(bld, rec, dist_matrix, "district", cfg)
    A_city,  R_city,  P_city  = _compute_level(bld, rec, dist_matrix, "city", cfg)
    bld[A_local.name] = A_local
    bld[A_dist.name]  = A_dist
    bld[A_city.name]  = A_city

    # интегральный показатель
    bld["A_h"] = (
        sl["local"]["weight"]    * A_local
        + sl["district"]["weight"] * A_dist
        + sl["city"]["weight"]     * A_city
    )

    # метрики по объектам
    rec["P_j"] = (
        pd.concat([P_local, P_dist, P_city])
        .groupby(level=0).max()
        .reindex(rec.index).fillna(0)
    )
    rec["R_j"] = (
        pd.concat([R_local, R_dist, R_city])
        .groupby(level=0).max()
        .reindex(rec.index).fillna(0)
    )

    # A_m = Σ C'_j / Σ P_j  (теоретический максимум обеспеченности)
    total_cap = float(rec["capacity_adj"].sum())
    total_pop = float(bld["population"].sum())
    A_m = total_cap / max(total_pop, 1.0)

    metrics: dict = {
        "n_buildings":       int(len(bld)),
        "n_recreation":      int(len(rec)),
        "population":        int(total_pop),
        "capacity_adj_total": float(total_cap),
        "A_h_mean":          float(bld["A_h"].mean()),
        "A_h_median":        float(bld["A_h"].median()),
        "A_h_min":           float(bld["A_h"].min()),
        "A_h_max":           float(bld["A_h"].max()),
        "A_m":               float(A_m),
        "CV":                float(bld["A_h"].std() / max(bld["A_h"].mean(), EPS)),
    }
    return bld, rec, metrics
