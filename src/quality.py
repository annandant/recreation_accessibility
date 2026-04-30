"""Качественные коэффициенты объектов рекреации (таблица 6 НИР).

Коэффициенты:
    k_zone   — соответствие функциональному зонированию
    k_ndvi   — растительный покров (NDVI)
    k_water  — близость к водоёмам
    k_road   — удалённость от автомагистралей

Скорректированная вместимость:
    C'_j = capacity_j × α_внеграницы × (1 + Σ w_k · (k_j − 1))
"""
from __future__ import annotations
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping

log = logging.getLogger(__name__)


# --- значения коэффициентов ---------------------------------
ZONE_LABEL_TO_COEF: dict[str, float] = {
    # Рекреационные зоны
    "р1":  1.20, "р2":  1.20, "р3":  1.20, "сп3": 1.20,
    # Жилые зоны
    "ж1":  1.00, "ж2":  1.00, "ж3":  1.00, "ж4":  1.00,
    # Общественно-деловые
    "д":   0.90, "см":  0.90,
    # Сельскохозяйственные
    "с1":  0.80, "с3":  0.80, "со":  0.80,
    # Инженерной и транспортной инфраструктуры
    "ти":  0.65, "и":   0.65,
    # Производственные
    "п":   0.60, "с2":  0.60,
    # Запретные / ограниченные для рекреации
    "сп1": 0.10, "рт":  0.10,
}

# Коэффициент для null-зон (акватория, незонированная территория)
_COEF_WATER_ZONE: float = 1.15
# Коэффициент-заглушка, если код не распознан
_COEF_UNKNOWN:    float = 1.00


def _zone_label_to_coef(label) -> float:
    """Код зоны → коэффициент k_zone.

    null / NaN / пустая строка → 1.15 (акватория или незонировано).
    Неизвестный код → 1.00 (нейтрально).
    """
    if label is None:
        return _COEF_WATER_ZONE
    try:
        import math
        if math.isnan(float(label)):        # pd.NA, np.nan, float('nan')
            return _COEF_WATER_ZONE
    except (TypeError, ValueError):
        pass
    s = str(label).strip().lower()
    if not s:
        return _COEF_WATER_ZONE
    coef = ZONE_LABEL_TO_COEF.get(s)
    if coef is not None:
        return coef
    log.debug("k_zone: неизвестный код зоны «%s» → %.2f", label, _COEF_UNKNOWN)
    return _COEF_UNKNOWN


# -------------------------------------------------------------------- #
#        Вспомогательные функции-пороговики (вынесены на уровень        #
#        модуля, чтобы их можно было протестировать отдельно)           #
# -------------------------------------------------------------------- #

def _ndvi_to_coef(v: float) -> float:
    """Таблица 6 НИР: NDVI → k_ndvi.

    Аргумент ``v`` должен быть в диапазоне [-1, 1] (стандартный NDVI).
    """
    if v > 0.66:
        return 1.30
    if v >= 0.33:
        return 1.00
    return 0.50


def _water_dist_to_coef(d: float) -> float:
    """Таблица 6 НИР: расстояние до водоёма (м) → k_water."""
    if d < 100:
        return 1.30
    if d <= 300:
        return 1.20
    if d <= 800:
        return 1.10
    return 1.00


def _road_dist_to_coef(d: float) -> float:
    """Таблица 6 НИР: расстояние до магистрали (м) → k_road."""
    if d > 500:
        return 1.00
    if d > 300:
        return 0.90
    if d > 150:
        return 0.80
    if d > 50:
        return 0.65
    return 0.50


# -------------------------------------------------------------------- #
#                      ОСНОВНЫЕ РАСЧЁТНЫЕ ФУНКЦИИ                      #
# -------------------------------------------------------------------- #

def k_zone(rec: gpd.GeoDataFrame, zoning: gpd.GeoDataFrame | None) -> pd.Series:
    """Коэффициент соответствия функциональному зонированию.

    Исправления по отношению к ранней версии:
    * CRS: зонирование явно приводится к CRS rec-слоя перед sjoin.
    * GeoDataFrame из центроидов: строится через gpd.GeoSeries с явным crs —
      избегает потери CRS при rec["geometry"] = Series (не GeoSeries).
    * predicate="intersects" вместо "within" — точки на границе зоны
      не падают в пустой результат.
    * fillna(_COEF_WATER_ZONE) в конце — объекты вне всех зон считаются
      акваторией (1.15), а не нейтральными (1.00).
    """
    if zoning is None or zoning.empty:
        return pd.Series(1.00, index=rec.index)

    # Поиск колонки без учёта регистра: LABEL, label, Label — всё одно
    _LABEL_CANDIDATES = {"label", "zone", "type", "тип"}
    label_col = next(
        (c for c in zoning.columns if c.lower() in _LABEL_CANDIDATES),
        None,
    )
    if label_col is None:
        log.warning(
            "k_zone: ни одна из колонок %s не найдена в слое зонирования. "
            "Имеющиеся колонки: %s — k_zone = 1.00",
            sorted(_LABEL_CANDIDATES), list(zoning.columns),
        )
        return pd.Series(1.00, index=rec.index)

    # ── CRS выравнивание ─────────────────────────────────────────────────
    zon = zoning if str(zoning.crs) == str(rec.crs) else zoning.to_crs(rec.crs)

    # ── Правильный GeoDataFrame из центроидов (CRS передаётся явно) ──────
    #    rec["centroid"] — обычная Series Shapely-точек, не GeoSeries.
    #    Прямое присвоение rec["geometry"] = rec["centroid"] теряет CRS
    #    в некоторых версиях GeoPandas → sjoin возвращает 0 совпадений.
    centroids = gpd.GeoDataFrame(
        index=rec.index,
        geometry=gpd.GeoSeries(list(rec["centroid"]), crs=rec.crs),
        crs=rec.crs,
    )

    joined = gpd.sjoin(
        centroids[["geometry"]],
        zon[[label_col, "geometry"]],
        how="left",
        predicate="intersects",   # "within" даёт пустой результат для точек на границе
    )

    n_matched = int(joined[label_col].notna().sum())
    log.info(
        "k_zone: sjoin совпадений %d / %d (колонка «%s»)",
        n_matched, len(rec), label_col,
    )
    if n_matched == 0:
        # Диагностика: сравниваем bbox объектов и зонирования
        rb = centroids.total_bounds   # (minx, miny, maxx, maxy)
        zb = zon.total_bounds
        _rec_center  = (f"({(rb[0]+rb[2])/2:.0f}, {(rb[1]+rb[3])/2:.0f})")
        _zone_center = (f"({(zb[0]+zb[2])/2:.0f}, {(zb[1]+zb[3])/2:.0f})")
        log.warning(
            "k_zone: 0 совпадений — объект(ы) вне покрытия зонирования "
            "(CRS=%s). Центр объектов: %s, центр зонирования: %s. "
            "k_zone = %.2f (фолбек). Задайте k_zone вручную в сценарии 3.",
            rec.crs, _rec_center, _zone_center, _COEF_WATER_ZONE,
        )

    coefs = joined[label_col].apply(_zone_label_to_coef)
    # max() выбирает наилучший коэффициент, если объект попадает в несколько зон.
    # fillna(_COEF_WATER_ZONE): объекты вне всех зон — акватория (1.15).
    return coefs.groupby(joined.index).max().reindex(rec.index).fillna(_COEF_WATER_ZONE)


def k_ndvi(rec: gpd.GeoDataFrame, ndvi_path: str | None) -> pd.Series:
    """Коэффициент растительного покрова (зональная статистика по NDVI).

    Если ``ndvi_path`` равен None, пустой строке или файл не существует —
    возвращает нейтральное значение k=1.00 для всех объектов.
    Предупреждает, если растр содержит не-отмасштабированные
    значения Landsat (средние > 1.5 до приведения к [-1, 1]).
    """
    if not ndvi_path:
        log.info("NDVI-растр не задан — k_ndvi = 1.00")
        return pd.Series(1.00, index=rec.index)
    try:
        src = rasterio.open(ndvi_path)
    except Exception as e:
        log.warning("NDVI недоступен: %s → k_ndvi = 1", e)
        return pd.Series(1.00, index=rec.index)

    rec_proj = rec.to_crs(src.crs)
    raw_means: list[float | None] = []
    unscaled_detected = False

    for geom in rec_proj.geometry:
        try:
            arr, _ = rio_mask(src, [mapping(geom)], crop=True, filled=False)
            vals = arr.compressed() if hasattr(arr, "compressed") else arr.flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                raw_means.append(None)
                continue
            mean = float(vals.mean())
            raw_means.append(mean)
        except Exception:
            raw_means.append(None)
    src.close()

    # Проверка масштаба ДО автокоррекции
    numeric = [v for v in raw_means if v is not None]
    if numeric and max(numeric) > 1.5:
        unscaled_detected = True
        log.warning(
            "NDVI: обнаружены значения > 1.5 (максимум = %.1f). "
            "Вероятно, Landsat-растр хранится в integer-единицах (× 0.0001) "
            "и не был отмасштабирован перед расчётом. "
            "Автоматически делим на 10 000 — проверьте источник данных.",
            max(numeric),
        )

    out: list[float | None] = []
    for mean in raw_means:
        if mean is None:
            out.append(None)
            continue
        if mean > 1.5:
            mean = mean / 10_000.0
        out.append(mean)

    s = pd.Series(
        [v if v is not None else float("nan") for v in out],
        index=rec.index,
    )
    return s.apply(lambda v: 1.00 if pd.isna(v) else _ndvi_to_coef(v)).rename("k_ndvi")


def k_water(rec: gpd.GeoDataFrame, water: gpd.GeoDataFrame) -> pd.Series:
    """Коэффициент близости к водоёмам."""
    if water is None or water.empty:
        return pd.Series(1.00, index=rec.index)
    rec_pts = gpd.GeoDataFrame(geometry=rec["centroid"], crs=rec.crs)
    water_union = water.unary_union
    dist = rec_pts.geometry.distance(water_union)
    return dist.apply(_water_dist_to_coef).rename("k_water")


def k_road(rec: gpd.GeoDataFrame, highway: gpd.GeoDataFrame) -> pd.Series:
    """Коэффициент удалённости от дорог (k_road).

    highway — GeoDataFrame, возвращённый load_motorway_edges():
    * если в нём есть магистральные типы (motorway/trunk/primary) —
      расстояние считается только до них;
    * если магистральных нет, но есть другие дороги — используются все
      (актуально для островных территорий без магистралей);
    * если highway пуст или None — k_road = 1.00 (нейтральное).
    """
    if highway is None or highway.empty:
        return pd.Series(1.00, index=rec.index)

    _MAJOR = {"motorway", "trunk", "primary",
              "motorway_link", "trunk_link", "primary_link"}

    def _is_major(v) -> bool:
        if isinstance(v, list):
            return bool(set(v) & _MAJOR)
        return str(v) in _MAJOR

    # Пробуем использовать только магистральные типы
    if "highway" in highway.columns:
        major = highway[highway["highway"].apply(_is_major)]
        roads = major if not major.empty else highway
    else:
        roads = highway

    if roads.empty:
        return pd.Series(1.00, index=rec.index)

    rec_pts = gpd.GeoDataFrame(geometry=rec["centroid"], crs=rec.crs)
    # приводим CRS дорог к CRS объектов (на случай расхождения)
    if roads.crs is not None and str(roads.crs) != str(rec.crs):
        roads = roads.to_crs(rec.crs)
    union = roads.unary_union
    dist = rec_pts.geometry.distance(union)
    return dist.apply(_road_dist_to_coef).rename("k_road")

def calculate_quality_factor(rec: gpd.GeoDataFrame, w: dict) -> pd.Series:
    """quality_factor = 1 + Σ w_k · (k_j − 1)."""
    return (
        1
        + w["zone"] * (rec["k_zone"] - 1)
        + w["ndvi"] * (rec["k_ndvi"] - 1)
        + w["water"] * (rec["k_water"] - 1)
        + w["road"] * (rec["k_road"] - 1)
    )


def apply_quality(
    recreation: gpd.GeoDataFrame,
    zoning: gpd.GeoDataFrame | None,
    water: gpd.GeoDataFrame,
    highway: gpd.GeoDataFrame,
    ndvi_path: str | None,
    cfg: dict,
) -> gpd.GeoDataFrame:
    """Записать k_zone, k_ndvi, k_water, k_road, quality_factor, capacity_adj.

    ``ndvi_path`` может быть None или пустой строкой — тогда k_ndvi = 1.00.
    """
    r = recreation.copy()
    r["k_zone"] = k_zone(r, zoning)
    r["k_ndvi"] = k_ndvi(r, ndvi_path)
    r["k_water"] = k_water(r, water)
    r["k_road"] = k_road(r, highway)
    r["quality_factor"] = calculate_quality_factor(r, cfg["ahp_weights"])
    r["capacity_adj"] = r["capacity"] * r["quality_factor"]

    # понижающий коэффициент α=0.5 для объектов вне границы (см. формула 1 НИР)
    alpha: float = cfg.get("external_alpha", 0.5)
    if "within_boundary" not in r.columns:
        r["within_boundary"] = True
    ext_mask = ~r["within_boundary"].fillna(True)
    n_ext = int(ext_mask.sum())
    if n_ext:
        r.loc[ext_mask, "capacity_adj"] *= alpha
        log.info("α=%.2f применён к %d внешним объектам", alpha, n_ext)
    return r

