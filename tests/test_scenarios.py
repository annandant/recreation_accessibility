"""Smoke-тесты сценариев: инфраструктурного и демографического.

Все тесты используют только синтетические данные — OSM и реальные файлы
не нужны.

Проверяемые инварианты
----------------------
Инфраструктурный сценарий (run_infrastructure_scenario):
    I-1. Добавление нового рекреационного объекта расширяет recreation (больше строк).
    I-2. Новый рекреационный объект появляется в dist_matrix хотя бы для одного здания.
    I-3. Baseline-пороги q_low / q_high сохраняются в metrics.
    I-4. modify несуществующего id не вызывает исключения (только WARNING).
    I-5. Пустой new_recr не вызывает исключения (только WARNING).
    I-6. deficit_share ∈ [0, 1].

Демографический сценарий (run_demographic_scenario):
    D-1. growth_pct увеличивает суммарное население.
    D-2. Новые здания добавляются в buildings_out.
    D-3. Новые здания без записей в dist_matrix → A_h = 0.
    D-4. Baseline-пороги q_low / q_high сохраняются в metrics.
    D-5. Пустой new_buildings не вызывает исключения.
"""
from __future__ import annotations
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

from src.scenarios import (
    run_baseline,
    run_infrastructure_scenario,
    run_demographic_scenario,
)


# -------------------------------------------------------------------- #
# Фабрика toy-данных
# -------------------------------------------------------------------- #

_CFG = {
    "service_levels": {
        "local":    {"radius": 400,  "area_max": 30_000,  "beta": 0.004,  "weight": 1.0},
        "district": {"radius": 1500, "area_max": 150_000, "beta": 0.0015, "weight": 0.0},
        "city":     {"radius": 6000, "area_max": 1e12,    "beta": 0.0004, "weight": 0.0},
    },
    "huff": {"lambda": 1.0},
    "deficit": {"q_low": 0.25, "q_high": 0.75},
    "capacity_per_ha": 100,
    "ahp_weights": {"zone": 0.263, "ndvi": 0.558, "water": 0.088, "road": 0.091},
    "external_alpha": 0.5,
    "crs": "EPSG:32636",
    "population": {
        "living_ratio": 0.85,
        "sqm_per_person": 28,
        "floor_search_radius": 300.0,
    },
}


def _make_square(cx: float, cy: float, side: float = 100.0) -> Polygon:
    h = side / 2
    return Polygon([
        (cx - h, cy - h), (cx + h, cy - h),
        (cx + h, cy + h), (cx - h, cy + h),
    ])


def _toy_buildings(n: int = 4) -> gpd.GeoDataFrame:
    """n жилых домов вдоль оси X с шагом 100 м, pop=50."""
    bld = gpd.GeoDataFrame(
        {
            "population": [50] * n,
            "floors":     [5.0] * n,
        },
        geometry=[Point(i * 100, 0) for i in range(n)],
        crs="EPSG:32636",
    )
    bld["centroid"] = bld.geometry
    return bld


def _toy_recreation() -> gpd.GeoDataFrame:
    """2 local-объекта рекреации, один внутри границы, один рядом."""
    geoms = [_make_square(0, 0, 200), _make_square(300, 0, 200)]
    rec = gpd.GeoDataFrame(
        {
            "capacity":      [20.0, 20.0],
            "capacity_adj":  [20.0, 20.0],
            "service_level": ["local", "local"],
            "within_boundary": [True, True],
            "k_zone":  [1.0, 1.0],
            "k_ndvi":  [1.0, 1.0],
            "k_water": [1.0, 1.0],
            "k_road":  [1.0, 1.0],
            "quality_factor": [1.0, 1.0],
            "area_m2": [40000.0, 40000.0],
        },
        geometry=geoms,
        crs="EPSG:32636",
    )
    rec["centroid"] = rec.geometry.centroid
    return rec


def _toy_dist_matrix(n_bld: int, n_rec: int) -> dict[int, dict[int, float]]:
    """Каждое здание видит оба рекреационных объекта на расстоянии 200 м."""
    return {
        b: {r: 200.0 for r in range(n_rec)}
        for b in range(n_bld)
    }


def _run_baseline():
    bld = _toy_buildings()
    rec = _toy_recreation()
    dm = _toy_dist_matrix(len(bld), len(rec))
    bld_out, rec_out, metrics, thresholds = run_baseline(bld, rec, dm, _CFG)
    return bld_out, rec_out, metrics, thresholds, bld, rec, dm


# -------------------------------------------------------------------- #
# Инфраструктурный сценарий
# -------------------------------------------------------------------- #

def test_infra_new_recr_added_to_recreation():
    """I-1: новый рекреационный объект увеличивает число объектов в recreation_out.

    G=None → dist_matrix не расширяется (только WARNING), но объект
    всё равно добавляется в таблицу recreation.
    """
    bld_b, rec_b, met_b, thr, bld, rec, dm = _run_baseline()

    new_recr = gpd.GeoDataFrame(
        {
            "service_level": ["local"],
            "area_m2": [5000.0],
        },
        geometry=[_make_square(200, 0, 100)],  # в пределах bbox зданий
        crs="EPSG:32636",
    )
    # G=None — dist_matrix не расширяется, но recreation должна вырасти
    bld_out, rec_out, _ = run_infrastructure_scenario(
        buildings=bld, recreation=rec, dist_matrix=dm,
        G=None, building_nodes=None,
        cfg=_CFG, baseline_thresholds=thr,
        new_recr=new_recr,
    )
    assert len(rec_out) > len(rec), (
        f"Ожидалось recreation_out > {len(rec)}, получено {len(rec_out)}"
    )


def test_infra_baseline_thresholds_preserved():
    """I-3: q_low и q_high в metrics совпадают с baseline."""
    bld_b, rec_b, met_b, thr, bld, rec, dm = _run_baseline()
    bld_out, rec_out, metrics = run_infrastructure_scenario(
        buildings=bld, recreation=rec, dist_matrix=dm,
        G=None, building_nodes=None,
        cfg=_CFG, baseline_thresholds=thr,
        new_recr=None,
        modify=None,
    )
    assert abs(metrics["q_low"]  - thr[0]) < 1e-9
    assert abs(metrics["q_high"] - thr[1]) < 1e-9


def test_infra_modify_nonexistent_id_no_crash():
    """I-4: modify с несуществующим id — без исключения."""
    bld_b, rec_b, met_b, thr, bld, rec, dm = _run_baseline()
    # id 9999 заведомо не существует
    run_infrastructure_scenario(
        buildings=bld, recreation=rec, dist_matrix=dm,
        G=None, building_nodes=None,
        cfg=_CFG, baseline_thresholds=thr,
        new_recr=None,
        modify={9999: {"k_ndvi": 1.30}},
    )


def test_infra_empty_new_recr_no_crash():
    """I-5: пустой new_recr — без исключения, только WARNING."""
    bld_b, rec_b, met_b, thr, bld, rec, dm = _run_baseline()
    empty = gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32636")
    run_infrastructure_scenario(
        buildings=bld, recreation=rec, dist_matrix=dm,
        G=None, building_nodes=None,
        cfg=_CFG, baseline_thresholds=thr,
        new_recr=empty,
    )


def test_infra_deficit_share_range():
    """I-6: deficit_share ∈ [0, 1]."""
    bld_b, rec_b, met_b, thr, bld, rec, dm = _run_baseline()
    _, _, metrics = run_infrastructure_scenario(
        buildings=bld, recreation=rec, dist_matrix=dm,
        G=None, building_nodes=None,
        cfg=_CFG, baseline_thresholds=thr,
    )
    assert 0.0 <= metrics["deficit_share"] <= 1.0


# -------------------------------------------------------------------- #
# Демографический сценарий
# -------------------------------------------------------------------- #

def test_demo_growth_increases_population():
    """D-1: growth_pct=20 увеличивает суммарное население."""
    bld_b, rec_b, met_b, thr, bld, rec, dm = _run_baseline()
    pop_before = int(bld["population"].sum())
    bld_out, _, _ = run_demographic_scenario(
        buildings=bld, recreation=rec, dist_matrix=dm,
        cfg=_CFG, baseline_thresholds=thr,
        growth_pct=20.0,
    )
    pop_after = int(bld_out["population"].sum())
    assert pop_after > pop_before, (
        f"После growth_pct=20 ожидалось population > {pop_before}, получено {pop_after}"
    )


def test_demo_new_buildings_added():
    """D-2: new_buildings добавляются в buildings_out."""
    bld_b, rec_b, met_b, thr, bld, rec, dm = _run_baseline()
    nb = gpd.GeoDataFrame(
        {"floors": [9.0]},
        geometry=[_make_square(500, 500, 50)],
        crs="EPSG:32636",
    )
    bld_out, _, _ = run_demographic_scenario(
        buildings=bld, recreation=rec, dist_matrix=dm,
        cfg=_CFG, baseline_thresholds=thr,
        new_buildings=nb,
    )
    assert len(bld_out) > len(bld), (
        f"Ожидалось buildings_out > {len(bld)}, получено {len(bld_out)}"
    )


def test_demo_new_buildings_zero_ah():
    """D-3: новые здания без записей в dist_matrix → A_h = 0."""
    bld_b, rec_b, met_b, thr, bld, rec, dm = _run_baseline()
    nb = gpd.GeoDataFrame(
        {"floors": [5.0]},
        geometry=[_make_square(9000, 9000, 50)],  # далеко, нет в dist_matrix
        crs="EPSG:32636",
    )
    bld_out, _, _ = run_demographic_scenario(
        buildings=bld, recreation=rec, dist_matrix=dm,
        cfg=_CFG, baseline_thresholds=thr,
        new_buildings=nb,
    )
    new_rows = bld_out.iloc[len(bld):]
    assert (new_rows["A_h"] == 0.0).all(), (
        f"Ожидалось A_h=0 для новых домов без dist_matrix, получено: {new_rows['A_h'].tolist()}"
    )


def test_demo_baseline_thresholds_preserved():
    """D-4: пороги из baseline сохраняются неизменными."""
    bld_b, rec_b, met_b, thr, bld, rec, dm = _run_baseline()
    _, _, metrics = run_demographic_scenario(
        buildings=bld, recreation=rec, dist_matrix=dm,
        cfg=_CFG, baseline_thresholds=thr,
        growth_pct=10.0,
    )
    assert abs(metrics["q_low"]  - thr[0]) < 1e-9
    assert abs(metrics["q_high"] - thr[1]) < 1e-9


def test_demo_empty_new_buildings_no_crash():
    """D-5: пустой new_buildings — без исключения."""
    bld_b, rec_b, met_b, thr, bld, rec, dm = _run_baseline()
    empty = gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:32636")
    run_demographic_scenario(
        buildings=bld, recreation=rec, dist_matrix=dm,
        cfg=_CFG, baseline_thresholds=thr,
        new_buildings=empty,
    )


# -------------------------------------------------------------------- #

if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  OK  · {t.__name__}")
        except Exception as e:
            failures += 1
            print(f"  FAIL · {t.__name__}: {e}")
    if failures == 0:
        print(f"\nВсе {len(tests)} тестов пройдены.")
    else:
        print(f"\nНЕ ПРОЙДЕНО: {failures} из {len(tests)}")
        sys.exit(1)



