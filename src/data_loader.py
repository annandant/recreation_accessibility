"""Загрузка и обрезка пространственных данных по выбранной территории.

Все слои приводятся к единой проекции, обрезаются по boundary с буфером
6000 м (максимальный радиус обслуживания) и обогащаются необходимыми
расчётными атрибутами.
"""
from __future__ import annotations
import logging
import pathlib
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

log = logging.getLogger(__name__)


# -------------------------------------------------------------------- #
#                       ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ                        #
# -------------------------------------------------------------------- #

def _safe_to_crs(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    return gdf.to_crs(crs) if str(gdf.crs) != crs else gdf


def _clip_with_buffer(
    layer: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    buffer_m: float = 6000.0,
) -> gpd.GeoDataFrame:
    """Обрезать layer полигоном boundary, расширенным на buffer_m."""
    boundary_buf = boundary.copy()
    boundary_buf["geometry"] = boundary.buffer(buffer_m)
    return gpd.clip(layer, boundary_buf)


def _validate_layer(
    gdf: gpd.GeoDataFrame,
    name: str,
    required_cols: set[str],
    allow_empty: bool = False,
) -> None:
    """Проверить наличие обязательных колонок и непустоту слоя.

    Поднимает ``ValueError`` с понятным сообщением вместо ``KeyError``
    при последующих обращениях к несуществующим колонкам.
    """
    if gdf.crs is None:
        raise ValueError(
            f"Слой «{name}»: отсутствует CRS. "
            "Установите проекцию в исходном файле или явно через to_crs()."
        )
    missing = required_cols - set(gdf.columns)
    if missing:
        raise ValueError(
            f"Слой «{name}»: отсутствуют обязательные колонки: "
            f"{sorted(missing)}. Имеющиеся колонки: {sorted(gdf.columns.tolist())}."
        )
    if not allow_empty and gdf.empty:
        raise ValueError(
            f"Слой «{name}» пуст после обрезки по территории. "
            "Проверьте исходный файл и соответствие CRS."
        )


# -------------------------------------------------------------------- #
#                           СТРОЕНИЯ                                   #
# -------------------------------------------------------------------- #

def _estimate_missing_floors(
    gdf: gpd.GeoDataFrame,
    search_radius: float = 300.0,
) -> pd.Series:
    """Пространственная импутация пропущенной этажности (трёхуровневый каскад).

    Алгоритм
    --------
    1. Источники этажности (в порядке приоритета):
         building:levels → B_LEVELS → height / 3.
    2. Для зданий с известной этажностью — значение сохраняется.
    3. Для неизвестных — **медиана соседей** в радиусе ``search_radius`` м
       (sjoin по расширенному буферу). Эксплуатирует пространственную
       автокорреляцию (первый закон Тоблера): здания одного квартала обычно
       строились в одну эпоху и имеют схожую типологию.
    4. Если в радиусе нет known-зданий — **глобальная медиана** территории
       (см. ниже).
    5. Если этажность неизвестна ни для одного здания — fallback = 2 эт.

    Почему именно медиана, а не среднее
    ------------------------------------
    Распределение этажности в российском городском фонде правостороннее
    (доминируют 5–9-этажные панельные дома, единичные высотки вытягивают
    хвост вправо). Медиана минимизирует MAE:

        θ̂_med = argmin_θ E[|X − θ|]

    Среднее минимизирует MSE, что оптимально при симметричном гауссовом
    распределении — здесь оно систематически завышает типичное значение
    из-за правого хвоста. Ошибка «+2 этажа» через цепочку
    living_area → population → P_j → R_j → A_h даёт смещение в оценках
    доступности соседних зданий.

    Почему «глобальная», а не IDW или расширение радиуса
    -----------------------------------------------------
    Глобальная медиана — непараметрический оценщик центра маргинального
    распределения при отсутствии пространственной информации. Она является
    предельным случаем расширения радиуса до ∞ и не требует
    гиперпараметра (показатель степени IDW, шаг расширения и т.п.).
    IDW с p=2 при расстоянии до ближайшего соседа > 500 м численно
    не отличается от глобального среднего, но вводит ложную точность
    через недетерминированный параметр p. Фиксированное значение (5 эт.)
    игнорирует наблюдаемые данные: при типичной застройке 9-этажками
    это занижает население на ~44 %.

    Практическая значимость fallback
    ---------------------------------
    В плотной городской застройке (Кронштадт, хорошее OSM-покрытие)
    уровень 4 срабатывает редко — только для одиночных периферийных
    зданий с пустым OSM-тегом. Их доля в суммарном населении мала,
    поэтому влияние на территориальные метрики S_T и CV незначительно.
    """
    floors = (
        pd.to_numeric(gdf.get("building:levels"), errors="coerce")
        if "building:levels" in gdf.columns
        else None
    )
    if floors is None and "B_LEVELS" in gdf.columns:
        floors = pd.to_numeric(gdf["B_LEVELS"], errors="coerce")
    if floors is None:
        floors = pd.Series(np.nan, index=gdf.index, dtype=float)

    if "height" in gdf.columns:
        h_floors = (pd.to_numeric(gdf["height"], errors="coerce") / 3.0).replace(0, np.nan)
        floors = floors.fillna(h_floors)

    known_mask = floors.notna() & (floors >= 1)
    unknown_mask = ~known_mask
    if known_mask.sum() == 0:
        # ничего не известно — fallback на 2 этажа
        return pd.Series(2.0, index=gdf.index)

    global_median = float(floors[known_mask].median())

    if "centroid" in gdf.columns:
        centroids = gdf["centroid"]
    else:
        centroids = gdf.geometry.centroid

    known_pts = gpd.GeoDataFrame(
        {"_fl": floors[known_mask].values},
        geometry=centroids[known_mask].values,
        crs=gdf.crs,
    )
    unknown_pts = gpd.GeoDataFrame(
        index=gdf[unknown_mask].index,
        geometry=centroids[unknown_mask].values,
        crs=gdf.crs,
    )
    if not unknown_pts.empty:
        unknown_buf = unknown_pts.copy()
        unknown_buf["geometry"] = unknown_pts.geometry.buffer(search_radius)
        joined = gpd.sjoin(
            unknown_buf[["geometry"]], known_pts[["geometry", "_fl"]],
            how="left", predicate="intersects",
        )
        neighbor_median = (
            joined.groupby(joined.index)["_fl"].median()
            .reindex(gdf[unknown_mask].index)
            .fillna(global_median)
        )
        floors = floors.copy()
        floors.loc[unknown_mask] = neighbor_median.values
    return floors.clip(lower=1)


def _is_living(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Жилое ли здание (см. таблицу 4 НИР).

    Приоритет источников:
    1. Колонка ``is_living`` (1 / True → жилое; NaN / 0 → нежилое).
       Важно: ``NaN`` трактуется как «нежилое» — именно для этого столбец
       и заводится в source-данных. Ошибка ``astype(bool)`` без
       предварительного ``fillna`` давала бы False-positive (nan → True).
    2. Тег ``BUILDING`` / ``building`` по классификатору OSM.
    3. Fallback — все здания считаются жилыми.
    """
    living_types = {
        "residential", "apartments", "house", "detached",
        "semidetached_house", "terrace", "dormitory", "allotment_house",
    }

    if "is_living" in gdf.columns:
        col = gdf["is_living"]
        if col.dtype == object:
            # строковые варианты: "True" / "1" / "да"
            true_vals = {"true", "1", "yes", "да"}
            mask = col.fillna("").astype(str).str.strip().str.lower().isin(true_vals)
        else:
            # float/int: NaN → 0 (нежилое), 1 → True
            mask = col.fillna(0).astype(bool)
        if mask.any():
            log.info(
                "Фильтр зданий (is_living): жилых %d из %d "
                "(нежилых/неразмеченных отсеяно: %d)",
                int(mask.sum()), len(mask), int((~mask).sum()),
            )
            return mask

    for col_name in ("BUILDING", "building"):
        if col_name in gdf.columns:
            mask = gdf[col_name].fillna("").astype(str).str.lower().isin(living_types)
            log.info(
                "Фильтр зданий (тег '%s'): жилых %d из %d",
                col_name, int(mask.sum()), len(mask),
            )
            return mask

    log.info("Фильтр зданий: жилая колонка не найдена — все %d зданий включены", len(gdf))
    return pd.Series(True, index=gdf.index)


def prepare_buildings_from_gkh(
    osm_buildings: gpd.GeoDataFrame,
    gkh_csv: str | pathlib.Path | None,
    living_ratio: float,
    sqm_per_person: float,
) -> gpd.GeoDataFrame:
    """Дополнить OSM-здания этажностью из АИС «Реформа ЖКХ» (если файл есть).

    Сопоставление идёт по адресу (улица + номер).

    Если ``gkh_csv`` равен None, пустой строке или файл не существует —
    возвращает исходный GeoDataFrame без изменений: этажность будет
    восстановлена статистически функцией ``_estimate_missing_floors``.
    """
    if not gkh_csv:
        log.info("Путь к АИС ЖКХ не задан — используем OSM/статистику этажности")
        return osm_buildings
    csv_path = pathlib.Path(gkh_csv)
    if not csv_path.exists():
        log.info("CSV АИС не найден: %s — используем OSM/статистику", csv_path.resolve())
        return osm_buildings

    try:
        ais = pd.read_csv(csv_path, sep=";", encoding="utf-8", low_memory=False)
    except Exception as e:
        log.warning("Не удалось прочитать %s: %s", csv_path, e)
        return osm_buildings

    if "address" not in ais.columns or "number_floors_max" not in ais.columns:
        log.warning(
            "В CSV АИС нет нужных колонок (address, number_floors_max) — "
            "пропускаем обогащение из ЖКХ."
        )
        return osm_buildings

    def norm_addr(s: object) -> str:
        return str(s).lower().replace(" ", "").replace(",", "").replace(".", "")

    # Берём только нужные колонки; переименовываем, чтобы не конфликтовали
    # с одноимёнными колонками в OSM-таблице зданий (иначе pandas добавит
    # суффиксы _x/_y и KeyError гарантирован).
    ais = ais[["address", "number_floors_max"]].copy()
    ais = ais.rename(columns={"number_floors_max": "_gkh_floors"})
    ais["_key"] = ais["address"].map(norm_addr)
    ais = ais.drop_duplicates("_key")

    bld = osm_buildings.copy()
    if "A_STRT" in bld.columns and "A_HSNMBR" in bld.columns:
        bld["_key"] = (bld["A_STRT"].fillna("") + bld["A_HSNMBR"].fillna("")).map(norm_addr)
    else:
        bld["_key"] = ""
    merged = bld.merge(ais[["_key", "_gkh_floors"]], on="_key", how="left")
    n_match = int(merged["_gkh_floors"].notna().sum())
    log.info("АИС ЖКХ: совпадений по адресу: %d / %d", n_match, len(merged))

    if "B_LEVELS" not in merged.columns:
        merged["B_LEVELS"] = np.nan
    merged["B_LEVELS"] = merged["B_LEVELS"].fillna(merged["_gkh_floors"])
    return merged.drop(columns=["_key", "_gkh_floors"], errors="ignore")


def prepare_buildings(
    raw: gpd.GeoDataFrame,
    cfg: dict,
) -> gpd.GeoDataFrame:
    """Финализация слоя зданий: фильтр жилых, оценка этажности, население."""
    gdf = raw.copy()
    n_raw = len(gdf)
    living_mask = _is_living(gdf)
    gdf = gdf[living_mask].reset_index(drop=True)
    n_excluded = n_raw - len(gdf)
    if n_excluded:
        log.info("Нежилых зданий отсеяно: %d (осталось %d)", n_excluded, len(gdf))

    gdf["centroid"] = gdf.geometry.centroid
    gdf["floors"] = _estimate_missing_floors(
        gdf, search_radius=cfg["population"]["floor_search_radius"]
    )
    gdf["footprint_area"] = gdf.geometry.area
    gdf["living_area"] = (
        gdf["footprint_area"] * gdf["floors"] * cfg["population"]["living_ratio"]
    )
    gdf["population"] = (
        gdf["living_area"] / cfg["population"]["sqm_per_person"]
    ).round(0).astype(int)
    return gdf


# -------------------------------------------------------------------- #
#                          РЕКРЕАЦИЯ                                   #
# -------------------------------------------------------------------- #

def prepare_recreation(
    raw: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    cfg: dict,
) -> gpd.GeoDataFrame:
    """Площадь, уровень обслуживания, нормативная вместимость."""
    gdf = raw.copy()
    gdf["area_m2"] = gdf.geometry.area
    gdf["centroid"] = gdf.geometry.centroid

    # уровень обслуживания по площади (таблица 5 НИР)
    sl = cfg["service_levels"]

    def _level(a: float) -> str:
        if a <= sl["local"]["area_max"]:
            return "local"
        if a <= sl["district"]["area_max"]:
            return "district"
        return "city"

    gdf["service_level"] = gdf["area_m2"].apply(_level)

    # нормативная вместимость (100 чел/га)
    gdf["capacity"] = gdf["area_m2"] / 1e4 * cfg["capacity_per_ha"]

    # принадлежит ли объект расчётной территории
    union = boundary.unary_union
    gdf["within_boundary"] = gdf.geometry.intersects(union)

    # Фильтр внешних объектов по радиусу их уровня обслуживания.
    # Внешний объект уровня L не может обслужить ни одно здание внутри
    # boundary, если расстояние от него до boundary > radius_L.
    # Это кратно сокращает число объектов в матрице расстояний.
    n_before = len(gdf)
    keep = gdf["within_boundary"].copy()           # внутренние — всегда оставляем
    for level, params in sl.items():
        radius = params["radius"]
        ext_level = (~gdf["within_boundary"]) & (gdf["service_level"] == level)
        if not ext_level.any():
            continue
        bnd_buf = boundary.buffer(radius).unary_union
        in_reach = gdf.loc[ext_level, "geometry"].intersects(bnd_buf)
        keep |= ext_level & in_reach.reindex(gdf.index, fill_value=False)
    gdf = gdf[keep].reset_index(drop=True)
    n_filtered = n_before - len(gdf)
    if n_filtered:
        log.info(
            "Внешних объектов отфильтровано (вне радиуса доступности): %d "
            "(осталось %d из %d)",
            n_filtered, len(gdf), n_before,
        )
    return gdf


# -------------------------------------------------------------------- #
#                  ЕДИНАЯ ТОЧКА ВХОДА — ЗАГРУЗКА ВСЕГО                 #
# -------------------------------------------------------------------- #

def load_all(cfg: dict, boundary: gpd.GeoDataFrame) -> dict:
    """Прочитать все слои, обрезать по territory+6000 м, привести к одной СК.

    Поднимает ``ValueError`` с понятным сообщением при отсутствии
    обязательных колонок или пустом слое.
    """
    crs: str = cfg["crs"]
    paths: dict = cfg["paths"]
    boundary = _safe_to_crs(boundary, crs)

    # область выборки (bbox)
    bbox_buf = boundary.buffer(6000).total_bounds
    bbox = box(*bbox_buf)

    def _read_layer(path: str) -> gpd.GeoDataFrame:
        """Прочитать слой с bbox-фильтром. Автоматически перепроецирует
        bbox в CRS файла — устойчиво к несовпадению СК между слоями."""
        # Шаг 1: определить CRS файла (1 строка = только метаданные)
        file_crs = None
        try:
            file_crs = gpd.read_file(path, rows=1).crs
        except Exception:
            pass

        # Шаг 2: перевести bbox в CRS файла для корректной фильтрации
        filter_bounds = bbox.bounds
        if file_crs is not None:
            try:
                bbox_native = gpd.GeoSeries([bbox], crs=crs).to_crs(file_crs).iloc[0]
                filter_bounds = bbox_native.bounds
            except Exception as _e:
                log.debug("bbox→CRS файла не преобразован (%s) — используем исходный", _e)

        gdf = gpd.read_file(path, bbox=filter_bounds)
        if gdf.crs is None:
            raise ValueError(
                f"Файл «{path}»: CRS не задан. "
                "Укажите проекцию в исходном файле."
            )
        return _safe_to_crs(gdf, crs)

    # --- recreation ---
    rec_raw = _read_layer(paths["recreation"])
    rec_raw = _clip_with_buffer(rec_raw, boundary, buffer_m=6000)
    _validate_layer(rec_raw, "recreation", required_cols=set())
    recreation = prepare_recreation(rec_raw, boundary, cfg)
    log.info(
        "Рекреационных объектов: %d (в т.ч. вне границы: %d)",
        len(recreation),
        int((~recreation["within_boundary"]).sum()),
    )

    # --- buildings (с обогащением из АИС ЖКХ) ---
    bld_raw = _read_layer(paths["building"])
    # клиппинг строго по territory (без буфера) — чтобы не считать чужое население
    _n_bld_in_bbox = len(bld_raw)
    bld_raw = gpd.clip(bld_raw, boundary)
    if bld_raw.empty:
        _bnd_bounds = boundary.total_bounds.round(0)
        if _n_bld_in_bbox == 0:
            raise ValueError(
                "Слой «building»: ни одного здания в bbox расчётной территории.\n"
                f"  Территория (X/Y): [{_bnd_bounds[0]}–{_bnd_bounds[2]}] / "
                f"[{_bnd_bounds[1]}–{_bnd_bounds[3]}]\n"
                "  Вероятная причина: building.gpkg покрывает другую область.\n"
                f"  Проверьте экстент файла {paths['building']} и убедитесь, "
                "что он содержит здания для выбранной территории."
            )
        raise ValueError(
            "Слой «building» пуст после обрезки по территории.\n"
            "  Проверьте геометрию boundary и building.gpkg."
        )
    _validate_layer(bld_raw, "building", required_cols=set())
    # gkh_csv опционален: None / "" → этажность из OSM + статистика соседей
    gkh_csv_path: str | None = paths.get("gkh_csv") or None
    bld_raw = prepare_buildings_from_gkh(
        bld_raw, gkh_csv_path,
        cfg["population"]["living_ratio"],
        cfg["population"]["sqm_per_person"],
    )
    buildings = prepare_buildings(bld_raw, cfg)
    if buildings.empty:
        raise ValueError(
            "После фильтрации жилых зданий список пуст. "
            "Проверьте атрибут 'building'/'BUILDING' в исходном файле."
        )
    log.info(
        "Жилых зданий: %d, расчётное население: %s",
        len(buildings),
        f"{int(buildings['population'].sum()):,}",
    )


    # --- water ---
    water = _read_layer(paths["water"])
    if water.empty:
        log.warning("Слой water пуст — k_water = 1.00 для всех объектов")

    # --- zoning (необязательный) ---
    zoning: Optional[gpd.GeoDataFrame]
    if pathlib.Path(paths["zoning"]).exists():
        zoning = _read_layer(paths["zoning"])
        if zoning.empty:
            log.warning("Слой zoning пуст — k_zone = 1.00 для всех объектов")
            zoning = None
    else:
        log.warning("Файл зонирования не найден: %s — k_zone = 1.00", paths["zoning"])
        zoning = None

    # --- NDVI raster (опциональный) ---
    # None / "" → k_ndvi = 1.00; файл не найден → то же самое + warning
    _ndvi_raw: str = paths.get("ndvi") or ""
    ndvi_path: str | None
    if not _ndvi_raw:
        log.info("NDVI-растр не задан в конфиге — k_ndvi = 1.00")
        ndvi_path = None
    elif not pathlib.Path(_ndvi_raw).exists():
        log.warning("NDVI-растр не найден: %s — k_ndvi = 1.00", _ndvi_raw)
        ndvi_path = None
    else:
        ndvi_path = _ndvi_raw

    return {
        "boundary": boundary,
        "recreation": recreation,
        "buildings": buildings,
        "water": water,
        "zoning": zoning,
        "ndvi_path": ndvi_path,   # str | None
    }



