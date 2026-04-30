"""Точка входа: каскадный выбор территории → baseline-анализ.

Запуск:
    python run.py                              # каскадный интерактив
    python run.py --territory "Кронштадт"      # без интерактива
    python run.py --territory "Парголово"
    python run.py --config myconfig.yaml       # альтернативный конфиг


"""
from __future__ import annotations
import argparse
import logging
import os
import pathlib
import sys

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
os.chdir(_SCRIPT_DIR)
sys.path.insert(0, str(_SCRIPT_DIR))

from src.config import load_config, setup_logging                   # noqa: E402
from src.territory import select_territory                           # noqa: E402
from src.data_loader import load_all                                 # noqa: E402
from src.quality import apply_quality                                # noqa: E402
from src.network import (                                            # noqa: E402
    load_or_download_graph, load_motorway_edges,
    build_node_index, build_distance_matrix,
)
from src.scenarios import run_baseline                               # noqa: E402
from src.export import export_results                                # noqa: E402
from src.visualization import (                                      # noqa: E402
    plot_quality_map, plot_accessibility_map,
    plot_deficit_map, plot_stress_map,
    plot_graph, plot_quality_metrics,
)

log = logging.getLogger(__name__)


def _safe_dirname(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _export_graph_gpkg(
    G,
    out_path: pathlib.Path,
    crs: str,
) -> None:
    """Экспорт рёбер и узлов пешеходной сети в GeoPackage.

    Слои:
        edges — LineString, атрибуты: length (м), highway, osmid
        nodes — Point,     атрибуты: osmid, x, y, street_count

    Включается флагом dev_export: true в config.yaml.
    """
    import osmnx as ox

    try:
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    except Exception as e:
        log.warning("_export_graph_gpkg: не удалось получить GDF из графа: %s", e)
        return

    # --- рёбра ---
    edge_cols = [c for c in ("osmid", "highway", "length") if c in edges_gdf.columns]
    edges_out = edges_gdf[edge_cols + ["geometry"]].copy()
    # highway может быть списком (osmnx объединяет параллельные пути)
    if "highway" in edges_out.columns:
        edges_out["highway"] = edges_out["highway"].apply(
            lambda v: v[0] if isinstance(v, list) else v
        )
    edges_out = edges_out.to_crs(crs).reset_index(drop=True)

    # --- узлы ---
    # osmid присутствует и в индексе, и в колонках — drop=True чтобы избежать
    # "cannot insert osmid, already exists" при to_file -> reset_index(drop=False)
    node_cols = [c for c in ("osmid", "x", "y", "street_count") if c in nodes_gdf.columns]
    nodes_out = nodes_gdf[node_cols + ["geometry"]].copy().to_crs(crs).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    edges_out.to_file(out_path, layer="edges", driver="GPKG")
    nodes_out.to_file(out_path, layer="nodes", driver="GPKG")
    log.info(
        "Граф экспортирован: %s  (рёбра: %d, узлы: %d)",
        out_path.name, len(edges_out), len(nodes_out),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Recreation accessibility")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--territory", default=None,
                   help="имя территории из NAME (минует интерактив)")
    args = p.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    # CRS: автоопределение из boundary.gpkg если не задан в config.yaml ----------
    if not cfg.get("crs"):
        import geopandas as _gpd
        _bnd_probe = _gpd.read_file(cfg["paths"]["boundary"], rows=1)
        _bnd_crs = _bnd_probe.crs
        if _bnd_crs is None:
            raise ValueError(
                "boundary.gpkg не содержит CRS. "
                "Задайте систему координат вручную: crs: \"EPSG:XXXXX\" в config.yaml"
            )
        if _bnd_crs.is_geographic:
            _cent = _bnd_probe.to_crs(4326).geometry.centroid.iloc[0]
            _zone = int((_cent.x + 180) / 6) + 1
            _epsg = (32600 if _cent.y >= 0 else 32700) + _zone
            cfg["crs"] = f"EPSG:{_epsg}"
            log.info("CRS не задан — автоматически выбрана UTM-зона: %s", cfg["crs"])
        else:
            cfg["crs"] = str(_bnd_crs)
            log.info("CRS не задан — определён из boundary.gpkg: %s", cfg["crs"])
    # ---------------------------------------------------------------------------

    # 1. ВЫБОР ТЕРРИТОРИИ ----------------------------------------------------
    boundary, territory_meta = select_territory(
        boundary_path=cfg["paths"]["boundary"],
        target_crs=cfg["crs"],
        territory_name=args.territory,
    )

    out_dir = pathlib.Path(cfg["paths"]["outputs_dir"]) / _safe_dirname(territory_meta["name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Результаты будут в: %s", out_dir)

    # 2. ЗАГРУЗКА ДАННЫХ -----------------------------------------------------
    log.info("[1/6] Загрузка данных…")
    data = load_all(cfg, boundary)

    # 3. ГРАФ + МАГИСТРАЛИ (OSM) ---------------------------------------------
    log.info("[2/6] Граф пешеходной сети и магистрали из OSM…")
    _network_type = cfg.get("network_type", "walk")
    _network_file = cfg["paths"].get("network_file", "") or None
    G = load_or_download_graph(
        boundary, cfg["crs"],
        network_type=_network_type,
        network_file=_network_file,
    )
    _highway_file = cfg["paths"].get("highway_file", "") or None
    highway_osm = load_motorway_edges(boundary, cfg["crs"], highway_file=_highway_file)

    # 4. КАЧЕСТВО ОБЪЕКТОВ ---------------------------------------------------
    log.info("[3/6] Расчёт качества рекреационных объектов…")
    recreation = apply_quality(
        data["recreation"], data["zoning"], data["water"], highway_osm,
        data["ndvi_path"], cfg,
    )

    # 5. МАТРИЦА РАССТОЯНИЙ --------------------------------------------------
    log.info("[4/6] Матрица расстояний (обратный Дейкстра)…")
    building_nodes = build_node_index(G, data["buildings"], geom_col="centroid")
    rec_nodes = build_node_index(G, recreation, geom_col="centroid")
    max_radius = max(s["radius"] for s in cfg["service_levels"].values())
    dist_matrix = build_distance_matrix(G, building_nodes, rec_nodes, max_radius)

    if cfg.get("dev_visuals", False):
        plot_graph(G, buildings=data["buildings"], recreation=recreation,
                   output_path=out_dir / "_dev_graph.png",
                   title=f"Пешеходная сеть · {territory_meta['name']}",
                   boundary=boundary)
        plot_quality_metrics(recreation,
                             output_path=out_dir / "_dev_quality_metrics.png",
                             title=f"Качество объектов · {territory_meta['name']}")

    if cfg.get("dev_export", False):
        _export_graph_gpkg(
            G,
            out_dir / "_dev_graph_network.gpkg",
            cfg["crs"],
        )

    # 5. БАЗОВЫЙ СЦЕНАРИЙ ----------------------------------------------------
    log.info("[5/6] Базовый сценарий: расчёт обеспеченности…")
    bld, rec, metrics, baseline_thresholds = run_baseline(
        data["buildings"], recreation, dist_matrix, cfg
    )
    log.info(
        "A_h: mean=%.3f, median=%.3f, CV=%.3f, A_m=%.3f, S_A=%.3f, deficit=%.1f%%",
        metrics["A_h_mean"], metrics["A_h_median"], metrics["CV"],
        metrics["A_m"], metrics["S_A"], metrics["deficit_share"] * 100,
    )

    # 6. КАРТЫ + ЭКСПОРТ -----------------------------------------------------
    log.info("[6/6] Сохранение карт и таблиц…")
    plot_quality_map(rec, out_dir / "map_quality.png",
                     title=f"Качество объектов · {territory_meta['name']}")
    _LEVEL_RU = {"local": "местная", "district": "районная", "city": "городская"}
    for level in ("local", "district", "city"):
        radius = cfg["service_levels"][level]["radius"]
        field = f"A_{radius}"
        plot_accessibility_map(bld, field, out_dir / f"map_accessibility_{level}.png",
                               title=f"Доступность {_LEVEL_RU[level]} ({radius} м) · {territory_meta['name']}")
    plot_accessibility_map(bld, "A_h", out_dir / "map_integral.png",
                           title=f"Интегральная обеспеченность A_h · {territory_meta['name']}")
    plot_deficit_map(bld, out_dir / "map_deficit.png",
                     title=f"Зоны дефицита · {territory_meta['name']}",
                     boundary=data["boundary"])
    plot_stress_map(bld, out_dir / "map_stress.png",
                    title=f"Рекреационный стресс S_i · {territory_meta['name']}")

    paths = export_results(bld, rec, metrics, out_dir, scenario_name="baseline")
    log.info("Готово. Результаты:")
    for k, v in paths.items():
        if v:
            log.info("  · %s: %s", k, v)
    log.info("Для сценарного анализа см. README.md (раздел «Сценарный анализ»).")


if __name__ == "__main__":
    main()
