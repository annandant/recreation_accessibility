"""Сценарии: базовый, инфраструктурный и демографический.

Все три функции принимают уже подготовленные слои из baseline-сценария 
— и возвращают пересчитанные buildings/recreation + метрики.
"""
from __future__ import annotations
import copy
import logging
from typing import Optional, Mapping

import geopandas as gpd
import pandas as pd

from .accessibility import calculate_accessibility
from .classify import classify_deficit_baseline, classify_with_thresholds
from .network import extend_distance_matrix
from .quality import apply_quality, calculate_quality_factor
from .stress import calculate_stress

log = logging.getLogger(__name__)


# -------------------------------------------------------------------- #
#                      ВНУТРЕННЯЯ ВАЛИДАЦИЯ                            #
# -------------------------------------------------------------------- #

def _validate_new_rec(
    gdf: gpd.GeoDataFrame,
    name: str,
    target_crs: str,
    ref_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Проверить и привести CRS нового GeoDataFrame рекреации/зданий.

    * CRS: если отличается от ``target_crs`` — перепроецируем.
    * Тип геометрии: ожидаются Polygon / MultiPolygon.
    * Пересечение с territory: проверяется по bounding-box зданий/рекреации.
    * Пустой GeoDataFrame: логируем предупреждение.
    """
    if gdf.empty:
        log.warning(
            "%s: передан пустой GeoDataFrame — операция не выполнена.", name
        )
        return gdf

    # --- CRS ---
    if gdf.crs is None:
        raise ValueError(
            f"{name}: отсутствует CRS. "
            "Установите систему координат перед передачей в сценарий."
        )
    if str(gdf.crs) != target_crs:
        log.info(
            "%s: CRS %s отличается от расчётного %s — перепроецируем.",
            name, gdf.crs, target_crs,
        )
        gdf = gdf.to_crs(target_crs)

    # --- тип геометрии ---
    bad_geom = [
        i for i, g in enumerate(gdf.geometry)
        if g is None or g.geom_type not in ("Polygon", "MultiPolygon")
    ]
    if bad_geom:
        raise ValueError(
            f"{name}: {len(bad_geom)} объект(ов) имеют не-полигональную "
            f"геометрию (позиции {bad_geom[:5]}). "
            "Ожидаются Polygon или MultiPolygon."
        )

    # --- пересечение с территорией (грубая проверка по bbox ref-слоя) ---
    if not ref_gdf.empty:
        territory_bbox = ref_gdf.total_bounds  # (minx, miny, maxx, maxy)
        outside = [
            i for i, g in enumerate(gdf.geometry)
            if (
                g.bounds[2] < territory_bbox[0] or g.bounds[0] > territory_bbox[2]
                or g.bounds[3] < territory_bbox[1] or g.bounds[1] > territory_bbox[3]
            )
        ]
        if outside:
            log.warning(
                "%s: %d объект(ов) полностью за пределами территории "
                "(позиции %s…). Проверьте CRS и координаты.",
                name, len(outside), outside[:5],
            )

    return gdf


# -------------------------------------------------------------------- #

def run_baseline(
    buildings: gpd.GeoDataFrame,
    recreation: gpd.GeoDataFrame,
    dist_matrix: dict,
    cfg: dict,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict, tuple[float, float]]:
    """Базовый сценарий. Возвращает (bld, rec, metrics, (q_low, q_high))."""
    bld, rec, metrics = calculate_accessibility(buildings, recreation, dist_matrix, cfg)
    bld, q_low, q_high = classify_deficit_baseline(bld, cfg)
    bld, S_A = calculate_stress(bld, metrics["A_m"])
    metrics["S_A"] = S_A
    metrics["q_low"] = q_low
    metrics["q_high"] = q_high
    metrics["deficit_share"] = float((bld["access_class"] == "deficit").mean())
    return bld, rec, metrics, (q_low, q_high)


# -------------------------------------------------------------------- #

def run_infrastructure_scenario(
    *,
    buildings: gpd.GeoDataFrame,
    recreation: gpd.GeoDataFrame,
    dist_matrix: dict,
    G,
    building_nodes: dict,
    cfg: dict,
    baseline_thresholds: tuple[float, float],
    new_recr: Optional[gpd.GeoDataFrame] = None,
    modify: Optional[Mapping[int, Mapping[str, float]]] = None,
    zoning: Optional[gpd.GeoDataFrame] = None,
    water: Optional[gpd.GeoDataFrame] = None,
    highway_osm: Optional[gpd.GeoDataFrame] = None,
    ndvi_path: Optional[str] = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict]:
    """Сценарий инфраструктурного развития.

    Параметры
    ---------
    new_recr : GeoDataFrame с гипотетическими объектами рекреации.
                Минимально нужны колонки `geometry` (Polygon) и
                `service_level` ∈ {local, district, city}; остальные
                атрибуты будут вычислены автоматически.
    modify    : словарь `{rec_index: {"k_zone": 1.2, "k_ndvi": 1.3, ...}}`,
                задающий новые значения коэффициентов качества для
                существующих объектов (например, благоустройство).

    Логика:
        1) В исходную таблицу recreation добавляются new_recr с расчётом
           area, capacity, capacity_adj, within_boundary.
        2) Для new_recr dist_matrix расширяется reverse-Dijkstra
           (без полного пересчёта).
        3) Применяются модификации `modify` (пересчитывается quality_factor
           и capacity_adj у указанных объектов).
        4) Расчёт A_h, классификация по фиксированным порогам, S_A.
    """
    rec = recreation.copy()
    base_idx = set(rec.index)
    sl = cfg["service_levels"]
    cap_per_ha = cfg["capacity_per_ha"]
    alpha = cfg.get("external_alpha", 0.5)

    # --- предупреждение при явно переданном пустом new_recr ------------
    if new_recr is not None and new_recr.empty:
        log.warning(
            "run_infrastructure_scenario: new_recr передан, но содержит 0 строк — "
            "новые объекты не будут добавлены."
        )

    # --- 1. добавление новых рекреационных объектов -------------------------------------
    if new_recr is not None and not new_recr.empty:
        nrec = _validate_new_rec(new_recr, "new_recr", cfg["crs"], buildings)

        if "centroid" not in nrec.columns:
            nrec["centroid"] = nrec.geometry.centroid
        if "area_m2" not in nrec.columns:
            nrec["area_m2"] = nrec.geometry.area
        if "service_level" not in nrec.columns:
            def _level(a: float) -> str:
                if a <= sl["local"]["area_max"]:
                    return "local"
                if a <= sl["district"]["area_max"]:
                    return "district"
                return "city"
            nrec["service_level"] = nrec["area_m2"].apply(_level)
        if "capacity" not in nrec.columns:
            nrec["capacity"] = nrec["area_m2"] / 1e4 * cap_per_ha
        # Коэффициенты качества: пересчитываем из слоёв данных, если они
        # переданы; иначе нейтральные (1.00). k_ndvi = 1.00 при отсутствии растра.
        _has_spatial = any(v is not None for v in (zoning, water, highway_osm))
        if _has_spatial or ndvi_path:
            # apply_quality полностью рассчитывает k_*, quality_factor,
            # capacity_adj и применяет alpha — повторный расчёт не нужен.
            nrec = apply_quality(nrec, zoning, water, highway_osm, ndvi_path, cfg)
        else:
            # Нейтральные коэффициенты → manual расчёт capacity_adj + alpha
            for col in ("k_zone", "k_ndvi", "k_water", "k_road"):
                if col not in nrec.columns:
                    nrec[col] = 1.00
            nrec["quality_factor"] = calculate_quality_factor(nrec, cfg["ahp_weights"])
            if "within_boundary" not in nrec.columns:
                nrec["within_boundary"] = True
            nrec["capacity_adj"] = nrec["capacity"] * nrec["quality_factor"]
            ext = ~nrec["within_boundary"].fillna(True)
            nrec.loc[ext, "capacity_adj"] *= alpha

        # назначаем индексы, не пересекающиеся с базовыми
        next_id = max(rec.index.max(), -1) + 1
        nrec.index = range(next_id, next_id + len(nrec))
        rec = pd.concat([rec, nrec])
        rec = gpd.GeoDataFrame(rec, geometry="geometry", crs=recreation.crs)
        new_idx = list(set(rec.index) - base_idx)
        log.info("Сценарий: добавлено новых рекреационных объектов: %d", len(new_idx))

        # расширяем dist_matrix (reverse Dijkstra от каждого нового парка)
        dist_matrix = copy.deepcopy(dist_matrix)
        rec_subset = rec.loc[new_idx].copy()
        max_radius = max(s["radius"] for s in sl.values())
        if G is not None and building_nodes:
            dist_matrix = extend_distance_matrix(
                G, building_nodes, rec_subset, max_radius, dist_matrix
            )
        else:
            log.warning(
                "run_infrastructure_scenario: G или building_nodes не переданы — "
                "dist_matrix для новых рекреационных объектов не расширяется. "
                "Новые объекты получат A_h = 0 для всех зданий."
            )

    # --- 2. модификация существующих объектов --------------------------
    if modify:
        for rid, changes in modify.items():
            if rid not in rec.index:
                log.warning("modify: id=%s нет в recreation — пропуск", rid)
                continue
            for k, v in changes.items():
                if k in rec.columns:
                    rec.at[rid, k] = float(v)
        # пересчитать quality_factor и capacity_adj для всех изменённых
        rec["quality_factor"] = calculate_quality_factor(rec, cfg["ahp_weights"])
        rec["capacity_adj"] = rec["capacity"] * rec["quality_factor"]
        ext = ~rec["within_boundary"].fillna(True)
        rec.loc[ext, "capacity_adj"] *= alpha
        log.info("Сценарий: изменено объектов: %d", len(modify))

    # --- 3. пересчёт обеспеченности и стресса --------------------------
    bld, rec_out, metrics = calculate_accessibility(buildings, rec, dist_matrix, cfg)
    q_low, q_high = baseline_thresholds
    bld = classify_with_thresholds(bld, q_low, q_high)
    bld, S_A = calculate_stress(bld, metrics["A_m"])
    metrics["S_A"] = S_A
    metrics["q_low"] = q_low
    metrics["q_high"] = q_high
    metrics["deficit_share"] = float((bld["access_class"] == "deficit").mean())
    return bld, rec_out, metrics


# -------------------------------------------------------------------- #

def run_demographic_scenario(
    *,
    buildings: gpd.GeoDataFrame,
    recreation: gpd.GeoDataFrame,
    dist_matrix: dict,
    cfg: dict,
    baseline_thresholds: tuple[float, float],
    new_buildings: Optional[gpd.GeoDataFrame] = None,
    growth_pct: float = 0.0,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict]:
    """Сценарий демографического развития"""
    bld = buildings.copy()

    if growth_pct:
        factor = 1.0 + growth_pct / 100.0
        bld["population"] = (bld["population"] * factor).round(0).astype(int)
        log.info("Сценарий: население существующих домов × %.3f", factor)

    new_idx_list: list[int] = []

    if new_buildings is not None and new_buildings.empty:
        log.warning(
            "run_demographic_scenario: new_buildings передан, но содержит 0 строк — "
            "новые здания не будут добавлены."
        )

    if new_buildings is not None and not new_buildings.empty:
        nb = _validate_new_rec(new_buildings, "new_buildings", cfg["crs"], bld)

        nb["centroid"] = nb.geometry.centroid

        if "population" in nb.columns and nb["population"].notna().any():
            # Численность уже задана в слое — используем напрямую.
            nb["population"] = nb["population"].fillna(0).round(0).astype(int)
            log.info(
                "Демосценарий: население новых зданий взято из атрибута 'population' "
                "(сумма: %s чел.)", f"{int(nb['population'].sum()):,}"
            )
        else:
            # Вычисляем из геометрии: площадь × этажность × доля жилья / норматив.
            if "floors" not in nb.columns:
                nb["floors"] = float(bld["floors"].median())
            nb["footprint_area"] = nb.geometry.area
            nb["living_area"] = (
                nb["footprint_area"] * nb["floors"] * cfg["population"]["living_ratio"]
            )
            nb["population"] = (
                nb["living_area"] / cfg["population"]["sqm_per_person"]
            ).round(0).astype(int)
            log.info(
                "Демосценарий: население новых зданий рассчитано по площади/этажности "
                "(сумма: %s чел.)", f"{int(nb['population'].sum()):,}"
            )
        next_id = max(bld.index.max(), -1) + 1
        nb.index = range(next_id, next_id + len(nb))
        new_idx_list = list(nb.index)
        bld = pd.concat([bld, nb])
        bld = gpd.GeoDataFrame(bld, geometry="geometry", crs=buildings.crs)
        log.info(
            "Сценарий: добавлено новых домов: %d, +население: %s",
            len(nb),
            f"{int(nb['population'].sum()):,}",
        )

    bld_out, rec_out, metrics = calculate_accessibility(bld, recreation, dist_matrix, cfg)
    q_low, q_high = baseline_thresholds
    bld_out = classify_with_thresholds(bld_out, q_low, q_high)
    bld_out, S_A = calculate_stress(bld_out, metrics["A_m"])

    # --- предупреждение: новые дома вне покрытия графа ------------------
    if new_idx_list:
        # Новые здания не имеют записей в dist_matrix → A_h останется 0
        # (это документированное поведение, но важно дать явный сигнал).
        new_bld_in_out = bld_out.loc[bld_out.index.isin(new_idx_list)]
        n_zero = int((new_bld_in_out["A_h"] == 0.0).sum())
        if n_zero:
            log.warning(
                "run_demographic_scenario: %d из %d новых зданий получили A_h = 0 "
                "(не покрыты графом пешеходной сети). "
                "Это нижняя оценка обеспеченности — см. docstring.",
                n_zero, len(new_idx_list),
            )

    metrics["S_A"] = S_A
    metrics["q_low"] = q_low
    metrics["q_high"] = q_high
    metrics["deficit_share"] = float((bld_out["access_class"] == "deficit").mean())
    return bld_out, rec_out, metrics


