"""Пешеходный граф OSM, привязка узлов и матрица расстояний.

Реализует:
    * load_or_download_graph    — кэшируемое получение пешеходной сети;
    * build_node_index           — отображение «объект → ближайший узел графа»;
    * build_distance_matrix      — матрица расстояний здание → объект;
    * extend_distance_matrix     — обратное Дейкстра для новых объектов.

Производительность:
    build_distance_matrix запускает single_source_dijkstra_path_length
    со стороны объектов рекреации (их обычно  меньше, чем зданий) 
    с cutoff=max_radius — это уже оптимальный «реверс-Дейкстра»
    подход. Батч-вариант через multi_source_dijkstra дал бы только
    минимальное расстояние от любого парка, но не на каждый объект
    в отдельности — поэтому он не применим для алгоритма 2SFCA.
"""
from __future__ import annotations
import logging
import pathlib

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import box

log = logging.getLogger(__name__)

ox.settings.use_cache = True
ox.settings.log_console = False


# -------------------------------------------------------------------- #
#                       ГРАФ ИЗ ЛОКАЛЬНОГО ФАЙЛА                       #
# -------------------------------------------------------------------- #

def _graph_from_lines(
    gdf: gpd.GeoDataFrame,
    crs: str,
    snap_tol: float = 0.5,
) -> nx.MultiDiGraph:
    """Построить osmnx-совместимый MultiDiGraph из GeoDataFrame с линиями.

    Каждая пара последовательных координат LineString образует направленное
    ребро (и обратное). Узлы получают атрибуты x/y, рёбра — length (метры).
    Близкие узлы (< snap_tol м) объединяются.

    Parameters
    ----------
    gdf       : GeoDataFrame с LineString / MultiLineString геометрией
    crs       : целевая метрическая CRS (EPSG:XXXXX)
    snap_tol  : точность привязки узлов в метрах (по умолчанию 0.5 м)
    """
    # Привести к метрической CRS
    if str(gdf.crs) != crs:
        gdf = gdf.to_crs(crs)

    # MultiLineString → LineString
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf = gdf[gdf.geometry.geom_type == "LineString"].copy()

    if gdf.empty:
        log.warning("_graph_from_lines: GeoDataFrame не содержит LineString — граф пуст")
        return nx.MultiDiGraph()

    # --- реестр узлов: снэп по сетке snap_tol ---
    _inv = 1.0 / snap_tol
    coord_to_id: dict[tuple, int] = {}
    id_to_coord: dict[int, tuple] = {}

    def _get_node(x: float, y: float) -> int:
        key = (round(x * _inv) / _inv, round(y * _inv) / _inv)
        if key not in coord_to_id:
            nid = len(coord_to_id)
            coord_to_id[key] = nid
            id_to_coord[nid] = key
        return coord_to_id[key]

    G = nx.MultiDiGraph()

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        coords = [c[:2] for c in geom.coords]  # убираем Z если есть
        for i in range(len(coords) - 1):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            u = _get_node(x0, y0)
            v = _get_node(x1, y1)
            if u == v:
                continue
            seg_len = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            # Двустороннее движение (пешеход)
            G.add_edge(u, v, key=0, length=seg_len)
            G.add_edge(v, u, key=0, length=seg_len)

    # Атрибуты узлов (x, y — нужны ox.distance.nearest_nodes).
    # Узлы из id_to_coord могут отсутствовать в G, если все их рёбра
    # были пропущены из-за u == v (снэп двух концов к одной точке).
    for nid, (x, y) in id_to_coord.items():
        if not G.has_node(nid):
            continue
        G.nodes[nid]["x"] = x
        G.nodes[nid]["y"] = y
        G.nodes[nid]["osmid"] = nid

    G.graph["crs"] = crs
    log.info(
        "_graph_from_lines: построен граф — %d узлов, %d рёбер",
        G.number_of_nodes(), G.number_of_edges(),
    )
    return G


# -------------------------------------------------------------------- #
#                       ЗАГРУЗКА / СКАЧИВАНИЕ ГРАФА                    #
# -------------------------------------------------------------------- #

def load_or_download_graph(
    boundary: gpd.GeoDataFrame,
    crs: str,
    network_type: str = "walk",
    network_file: str | None = None,
    cache_dir: str | pathlib.Path = "outputs/cache",
) -> nx.MultiDiGraph:
    """Получить пешеходный граф для расширенной территории.

    Приоритет источников:
      1. network_file — локальный .gpkg с LineString дорогами (если задан и существует).
         Граф строится функцией _graph_from_lines; полезен, когда OSM Walk не покрывает
         все улицы (напр. неразмеченные дороги).
      2. OSM — скачивается через osmnx.graph_from_bbox и кэшируется в outputs/cache/.
         network_type ("walk" / "all") контролирует тип сети.

    Parameters
    ----------
    boundary      : GeoDataFrame с границей территории
    crs           : метрическая CRS (EPSG:XXXXX)
    network_type  : тип OSM-сети ("walk" / "all"); "all" даёт лучшее покрытие
    network_file  : путь к локальному .gpkg; если None или файл не найден — OSM
    cache_dir     : каталог кэша GraphML
    """
    # --- Вариант 1: локальный файл дорог ------------------------------------
    if network_file:
        nf_path = pathlib.Path(network_file)
        if nf_path.exists():
            log.info("Граф: строим из локального файла '%s'...", nf_path.name)
            # Читаем только bbox территории + 6 км, иначе весь город = миллион узлов
            buf = 6000.0
            bnd_proj = (
                boundary.to_crs(crs) if str(boundary.crs) != crs else boundary
            )
            minx_p, miny_p, maxx_p, maxy_p = bnd_proj.total_bounds
            bbox_proj = (
                minx_p - buf, miny_p - buf,
                maxx_p + buf, maxy_p + buf,
            )
            try:
                _file_crs = gpd.read_file(nf_path, rows=1).crs
                if _file_crs and str(_file_crs) != crs:
                    _bbox_gs = gpd.GeoSeries(
                        [box(*bbox_proj)], crs=crs
                    ).to_crs(_file_crs)
                    bbox_read = _bbox_gs.iloc[0].bounds
                else:
                    bbox_read = bbox_proj
            except Exception:
                bbox_read = bbox_proj
            hw_gdf = gpd.read_file(nf_path, bbox=bbox_read)
            # приводим имена колонок к нижнему регистру для единообразия
            hw_gdf.columns = [c.lower() for c in hw_gdf.columns]
            G = _graph_from_lines(hw_gdf, crs)
            if G.number_of_nodes() > 0:
                log.info(
                    "Граф из файла: %d узлов, %d рёбер (bbox + %d м)",
                    G.number_of_nodes(), G.number_of_edges(), int(buf),
                )
                return G
            log.warning("Граф из файла пустой — переключаемся на OSM")
        else:
            log.warning(
                "network_file указан, но файл не найден: %s — переключаемся на OSM",
                nf_path,
            )
    # --- Вариант 2: OSM (с кэшированием) ------------------------------------
    cache = pathlib.Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # Буфер 6 000 м в метрической проекции, затем конвертируем bbox в WGS84
    buf = 6000.0
    bnd_proj = boundary.to_crs(crs) if str(boundary.crs) != crs else boundary
    minx_p, miny_p, maxx_p, maxy_p = bnd_proj.total_bounds
    expanded = gpd.GeoDataFrame(
        geometry=[box(minx_p - buf, miny_p - buf, maxx_p + buf, maxy_p + buf)],
        crs=crs,
    )
    bb = expanded.to_crs("EPSG:4326").total_bounds   # (W, S, E, N)
    minx, miny, maxx, maxy = bb

    fname = cache / (
        f"graph_{round(minx,3)}_{round(miny,3)}"
        f"_{round(maxx,3)}_{round(maxy,3)}.graphml"
    )
    if fname.exists():
        log.info("Граф: загружаем из кэша: %s", fname)
        G = ox.load_graphml(fname)
    else:
        log.info("Граф: скачиваем из OSM (network_type=%s)...", network_type)
        try:
            G = ox.graph_from_bbox(
                (minx, miny, maxx, maxy),
                network_type=network_type,
                simplify=True,
            )
        except Exception as _osm_err:
            raise ConnectionError(
                f"Не удалось скачать граф из OSM: {_osm_err}\n\n"
                "Нет подключения к интернету или Overpass API недоступен.\n"
                "Решения:\n"
                "  1. Подключитесь к интернету и запустите снова"
                " (граф закэшируется в outputs/cache/).\n"
                "  2. Укажите локальный файл дорог в config.yaml:\n"
                "       network_file: \"data/highway.gpkg\""
            ) from _osm_err
        ox.save_graphml(G, fname)
        log.info("Граф сохранён: %s", fname)

    G = ox.project_graph(G, to_crs=crs)
    return G


def load_motorway_edges(
    boundary: gpd.GeoDataFrame,
    crs: str,
    highway_file: str | None = None,
    cache_dir: str | pathlib.Path = "outputs/cache",
) -> gpd.GeoDataFrame:
    """Загрузить дороги для k_road.

    Приоритет источников:
    1. Локальный файл ``highway_file`` (если задан и существует).
       Сначала ищутся магистральные типы (motorway/trunk/primary);
       при их отсутствии возвращаются ВСЕ дороги файла — это позволяет
       оценивать удалённость даже на территориях без магистралей.
    2. Кэш OSM (outputs/cache/motorways_*.gpkg).
    3. Скачать из OSM (motorway/trunk/primary, буфер 6 000 м).

    Если ни один источник не дал результата — возвращает пустой
    GeoDataFrame; k_road вернёт 1.00 (нейтральное значение).
    """
    _MAJOR = {"motorway", "trunk", "primary",
              "motorway_link", "trunk_link", "primary_link"}
    _TAGS = list(_MAJOR)

    # -- 1. Локальный файл -------------------------------------------------
    if highway_file:
        hf = pathlib.Path(highway_file)
        if hf.exists():
            log.info("k_road: дороги из локального файла '%s'", hf.name)
            # Читаем только bbox территории + 6 км (файл может быть на весь город)
            buf = 6000.0
            bnd_p = boundary.to_crs(crs) if str(boundary.crs) != crs else boundary
            x0, y0, x1, y1 = bnd_p.total_bounds
            bbox_proj = (x0 - buf, y0 - buf, x1 + buf, y1 + buf)
            try:
                _fcrs = gpd.read_file(hf, rows=1).crs
                if _fcrs and str(_fcrs) != crs:
                    _bgs = gpd.GeoSeries([box(*bbox_proj)], crs=crs).to_crs(_fcrs)
                    bbox_read = _bgs.iloc[0].bounds
                else:
                    bbox_read = bbox_proj
                hw_raw = gpd.read_file(hf, bbox=bbox_read)
            except Exception as e:
                log.warning(
                    "k_road: не удалось прочитать %s: %s -> пробуем OSM", hf, e
                )
                hw_raw = None

            if hw_raw is not None and not hw_raw.empty:
                # нормализуем имена колонок (HIGHWAY -> highway и т.п.)
                hw_raw.columns = [c.lower() for c in hw_raw.columns]
                if hw_raw.crs is None:
                    log.warning("k_road: файл %s без CRS -> пробуем OSM", hf.name)
                else:
                    hw_raw = (
                        hw_raw.to_crs(crs) if str(hw_raw.crs) != crs else hw_raw
                    )
                    hw_raw = hw_raw[
                        hw_raw.geometry.geom_type.isin(
                            ["LineString", "MultiLineString"]
                        )
                    ].copy()
                    if not hw_raw.empty:
                        if "highway" in hw_raw.columns:
                            hw_raw["highway"] = hw_raw["highway"].apply(
                                lambda v: v[0] if isinstance(v, list) else v
                            )
                            major = hw_raw[
                                hw_raw["highway"].isin(_MAJOR)
                            ][["highway", "geometry"]].reset_index(drop=True)
                            if not major.empty:
                                log.info(
                                    "k_road: %d магистральных дорог из файла",
                                    len(major),
                                )
                                return major
                            # нет магистральных -> все дороги файла
                            log.info(
                                "k_road: магистральных дорог в файле нет "
                                "-> используем все дороги (%d) из '%s'",
                                len(hw_raw), hf.name,
                            )
                            return hw_raw[
                                ["highway", "geometry"]
                            ].reset_index(drop=True)
                        else:
                            # нет колонки highway -> все линии с placeholder
                            log.info(
                                "k_road: нет колонки highway -> "
                                "все линии (%d) из '%s'",
                                len(hw_raw), hf.name,
                            )
                            hw_raw = hw_raw.copy()
                            hw_raw["highway"] = "road"
                            return hw_raw[
                                ["highway", "geometry"]
                            ].reset_index(drop=True)
        else:
            log.warning(
                "k_road: highway_file '%s' не найден -> пробуем OSM", hf
            )
    # -- 2 + 3. OSM (кэш или скачать) --------------------------------------
    cache = pathlib.Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    buf = 6000.0
    bnd_proj = boundary.to_crs(crs) if str(boundary.crs) != crs else boundary
    minx_p, miny_p, maxx_p, maxy_p = bnd_proj.total_bounds
    expanded = gpd.GeoDataFrame(
        geometry=[box(minx_p - buf, miny_p - buf, maxx_p + buf, maxy_p + buf)],
        crs=crs,
    )
    bb = expanded.to_crs("EPSG:4326").total_bounds   # (W, S, E, N)
    minx, miny, maxx, maxy = bb

    fname = cache / (
        f"motorways_{round(minx, 3)}_{round(miny, 3)}"
        f"_{round(maxx, 3)}_{round(maxy, 3)}.gpkg"
    )
    if fname.exists():
        log.info("k_road (OSM): загружаем из кэша: %s", fname.name)
        hw = gpd.read_file(fname)
        return hw.to_crs(crs) if str(hw.crs) != crs else hw

    log.info("k_road (OSM): скачиваем motorway/trunk/primary...")
    try:
        features = ox.features_from_bbox(
            (minx, miny, maxx, maxy),
            tags={"highway": _TAGS},
        )
        hw = features[
            features.geometry.geom_type.isin(["LineString", "MultiLineString"])
        ].copy()
        if "highway" not in hw.columns or hw.empty:
            log.info("k_road (OSM): магистральных дорог в bbox не найдено")
            return gpd.GeoDataFrame()
        hw["highway"] = hw["highway"].apply(
            lambda v: v[0] if isinstance(v, list) else v
        )
        hw = (
            hw[hw["highway"].isin(_MAJOR)][["highway", "geometry"]]
            .reset_index(drop=True)
        )
        if hw.empty:
            log.info("k_road (OSM): магистральных дорог в bbox не найдено")
            return gpd.GeoDataFrame()
        hw = hw.to_crs(crs)
        hw.to_file(fname, driver="GPKG")
        log.info(
            "k_road (OSM): сохранено %d объектов -> %s", len(hw), fname.name
        )
    except Exception as e:
        log.warning(
            "k_road (OSM): не удалось скачать: %s -> k_road = 1.00", e
        )
        return gpd.GeoDataFrame()
    return hw
def build_node_index(
    G: nx.MultiDiGraph,
    gdf: gpd.GeoDataFrame,
    geom_col: str = "centroid",
) -> dict[int, int]:
    """Сопоставить каждому объекту ближайший узел графа."""
    geoms = gdf[geom_col] if geom_col in gdf.columns else gdf.geometry.centroid
    xs = [p.x for p in geoms]
    ys = [p.y for p in geoms]
    nodes = ox.distance.nearest_nodes(G, X=xs, Y=ys)
    return dict(zip(gdf.index.tolist(), nodes))


# -------------------------------------------------------------------- #

def build_distance_matrix(
    G: nx.MultiDiGraph,
    building_nodes: dict[int, int],
    rec_nodes: dict[int, int],
    max_radius: float,
) -> dict[int, dict[int, float]]:
    """Построить матрицу: dist_matrix[building_idx][rec_idx] = метры.

    Используется обход single-source Dijkstra со стороны объектов рекреации с cutoff=max_radius.
    """
    G_undir = G.to_undirected()
    node_to_buildings: dict[int, list[int]] = {}
    for b_idx, n in building_nodes.items():
        node_to_buildings.setdefault(n, []).append(b_idx)

    dist_matrix: dict[int, dict[int, float]] = {b: {} for b in building_nodes.keys()}
    n_rec = len(rec_nodes)
    for i, (r_idx, r_node) in enumerate(rec_nodes.items(), start=1):
        if i % 50 == 0 or i == n_rec:
            log.debug("dist_matrix: объект %d / %d", i, n_rec)
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G_undir, r_node, cutoff=max_radius, weight="length"
            )
        except (nx.NetworkXError, nx.NodeNotFound):
            log.warning("dist_matrix: узел %s (rec_idx=%s) не найден в графе", r_node, r_idx)
            continue
        for reached_node, dist in lengths.items():
            for b_idx in node_to_buildings.get(reached_node, ()):
                dist_matrix[b_idx][r_idx] = dist
    return dist_matrix


def extend_distance_matrix(
    G: nx.MultiDiGraph,
    building_nodes: dict[int, int],
    new_rec_gdf: gpd.GeoDataFrame,
    max_radius: float,
    dist_matrix: dict[int, dict[int, float]],
) -> dict[int, dict[int, float]]:
    """Дописать в существующую матрицу расстояния до новых объектов.

    Применяется в сценарии «новый объект» — без пересчёта по всем зданиям.
    """
    new_rec_nodes = build_node_index(G, new_rec_gdf, geom_col="centroid")
    node_to_buildings: dict[int, list[int]] = {}
    for b_idx, n in building_nodes.items():
        node_to_buildings.setdefault(n, []).append(b_idx)

    G_undir = G.to_undirected()
    for r_idx, rec_node in new_rec_nodes.items():
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G_undir, rec_node, cutoff=max_radius, weight="length"
            )
        except (nx.NetworkXError, nx.NodeNotFound):
            log.warning("extend_matrix: узел %s (rec_idx=%s) не найден в графе", rec_node, r_idx)
            continue
        for reached_node, dist in lengths.items():
            for b_idx in node_to_buildings.get(reached_node, ()):
                if b_idx in dist_matrix:
                    dist_matrix[b_idx][r_idx] = dist
    log.info("dist_matrix расширена для %d новых объектов", len(new_rec_nodes))
    return dist_matrix