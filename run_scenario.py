"""Интерактивный сценарный анализ.

Запуск:
    python run_scenario.py                         # интерактивный выбор территории
    python run_scenario.py --territory "Кронштадт" # без интерактива
    python run_scenario.py --territory "Парголово"
    python run_scenario.py --config myconfig.yaml  # альтернативный конфиг

Меню сценариев
--------------
1) Добавить новый рекреационный объект — из файла data/
2) Добавить новое жилое здание        — из файла data/
3) Улучшить существующий объект       — показывает карту с индексами, запрашивает k_zone и k_ndvi
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

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import contextily as ctx
from matplotlib.colors import LinearSegmentedColormap

from src.config import load_config, setup_logging
from src.territory import select_territory
from src.data_loader import load_all
from src.quality import apply_quality
from src.network import (
    load_or_download_graph, load_motorway_edges,
    build_node_index, build_distance_matrix,
)
from src.scenarios import (
    run_baseline,
    run_infrastructure_scenario,
    run_demographic_scenario,
)
from src.export import export_results, export_comparison
from src.visualization import (
    plot_deficit_map, plot_accessibility_map, CMAP_QUALITY,
)

log = logging.getLogger(__name__)

# -------------------------------------------------------------------- #
#                        ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ                       #
# -------------------------------------------------------------------- #

def _safe_dirname(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _ask(prompt: str, default=None, cast=str, valid: list | None = None):
    """Запросить ввод с проверкой типа и допустимых значений"""
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return cast(default) if cast is not str else default
        try:
            val = cast(raw)
        except (ValueError, TypeError):
            print(f"  ✗ Ожидается {cast.__name__}. Повторите ввод.")
            continue
        if valid is not None and val not in valid:
            print(f"  ✗ Допустимые значения: {', '.join(str(v) for v in valid)}")
            continue
        return val


def _open(path: pathlib.Path) -> None:
    """Открыть файл стандартным приложением Windows."""
    try:
        os.startfile(str(path))
    except Exception:
        pass


def _basemap(ax, crs) -> None:
    try:
        ctx.add_basemap(ax, crs=str(crs),
                        source=ctx.providers.CartoDB.PositronNoLabels, zorder=0)
    except Exception:
        pass


def _annotate_indices(ax, rec_wm: gpd.GeoDataFrame) -> None:
    """Нанести индексы на центроиды объектов."""
    for idx, row in rec_wm.iterrows():
        c = row.geometry.centroid
        ax.annotate(
            str(idx), xy=(c.x, c.y),
            fontsize=1, ha="center", va="center",
            fontweight="bold", color="#111111",
            zorder=6,
        )


def _save_map(fig, path: pathlib.Path, dpi: int = 200, show: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Карта: {path.name}")
    if show:
        _open(path)


# ── Карта качества с индексами ──────────────────────────────────────────

def _map_quality_indexed(rec: gpd.GeoDataFrame,
                         boundary: gpd.GeoDataFrame,
                         out_path: pathlib.Path,
                         title: str = "Качество объектов",
                         show: bool = False) -> None:
    """Карта quality_factor с цветовой шкалой"""
    rec_wm = rec.to_crs(epsg=3857)
    bnd_wm = boundary.to_crs(epsg=3857)

    vmin = float(rec_wm["quality_factor"].min())
    vmax = float(rec_wm["quality_factor"].max())
    if vmin == vmax:
        vmax = vmin + 0.01

    fig, ax = plt.subplots(figsize=(14, 10))
    rec_wm.plot(
        column="quality_factor", cmap=CMAP_QUALITY,
        vmin=vmin, vmax=vmax,
        alpha=0.88, legend=False, ax=ax, zorder=3,
    )
    bnd_wm.boundary.plot(ax=ax, color="#444444", linewidth=0.9, zorder=4)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    _basemap(ax, rec_wm.crs)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(title, pad=8)
    ax.set_axis_off()

    # Цветовая шкала справа
    sm = mpl.cm.ScalarMappable(
        cmap=CMAP_QUALITY,
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, shrink=0.80, aspect=28)
    cb.set_label("Коэффициент качества Q", fontsize=10, labelpad=10)
    cb.ax.tick_params(labelsize=9)
    ticks = np.linspace(vmin, vmax, 5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.3g}" for t in ticks])

    _save_map(fig, out_path, show=show)


# ── Превью: новые объекты поверх существующих ──────────────────────────

def _map_preview_new_recr(rec_existing: gpd.GeoDataFrame,
                           new_recr: gpd.GeoDataFrame,
                           boundary: gpd.GeoDataFrame,
                           out_path: pathlib.Path,
                           show: bool = False) -> None:
    """Превью: существующие объекты серые, новые — зелёные."""
    ex_wm  = rec_existing.to_crs(epsg=3857)
    new_wm = new_recr.to_crs(epsg=3857)
    bnd_wm = boundary.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(14, 10))
    ex_wm.plot(ax=ax,  color="#b0c4de", alpha=0.70, zorder=3)
    new_wm.plot(ax=ax, color="#2ca02c", alpha=0.90, zorder=4)
    bnd_wm.boundary.plot(ax=ax, color="#444444", linewidth=0.9, zorder=5)

    # Индексы существующих

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    _basemap(ax, ex_wm.crs)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title("Превью: новые объекты (зелёный) + существующие (синий)", pad=8)
    ax.set_axis_off()

    legend_handles = [
        mpatches.Patch(color="#b0c4de", label="Существующие объекты"),
        mpatches.Patch(color="#2ca02c", label="Новые объекты (из файла)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
              fontsize=10, framealpha=0.95, edgecolor="#cccccc")

    _save_map(fig, out_path, show=show)


# ── Превью: выбранный объект подсвечен ────────────────────────────────

def _map_preview_highlight(rec: gpd.GeoDataFrame,
                            obj_idx: int,
                            boundary: gpd.GeoDataFrame,
                            out_path: pathlib.Path,
                            show: bool = False) -> None:
    """Превью: все объекты серые, выбранный — зеленый с жирной границей."""
    rec_wm = rec.to_crs(epsg=3857)
    bnd_wm = boundary.to_crs(epsg=3857)

    other = rec_wm[rec_wm.index != obj_idx]
    sel   = rec_wm[rec_wm.index == obj_idx]

    fig, ax = plt.subplots(figsize=(14, 10))
    other.plot(ax=ax, color="#b0c4de", alpha=0.65, zorder=3)
    sel.plot(ax=ax,   color="#45ad38", alpha=0.92, zorder=4)
    sel.boundary.plot(ax=ax, color="#3C633C", linewidth=2.5, zorder=5)
    bnd_wm.boundary.plot(ax=ax, color="#444444", linewidth=0.9, zorder=4)


    # Дополнительная метка выбранного объекта
    c = sel.geometry.centroid.iloc[0]


    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    _basemap(ax, rec_wm.crs)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(f"Превью: выбранный объект {obj_idx} (зеленый)", pad=8)
    ax.set_axis_off()

    legend_handles = [
        mpatches.Patch(color="#b0c4de", label="Остальные объекты"),
        mpatches.Patch(color="#3C633C", label=f"Выбранный объект [{obj_idx}]"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
              fontsize=10, framealpha=0.95, edgecolor="#cccccc")

    _save_map(fig, out_path, show=show)


# ── Таблица объектов ────────────────────────────────────────────────────

def _print_recr_table(rec: gpd.GeoDataFrame) -> None:
    print()
    print(f"  {'Индекс':>7}  {'Уровень':<10}  {'Площадь, га':>11}  "
          f"{'k_zone':>7}  {'k_ndvi':>7}  {'k_water':>8}  {'k_road':>7}  {'Q':>7}")
    print("  " + "─" * 72)
    for idx, row in rec[["service_level", "area_m2",
                          "k_zone", "k_ndvi", "k_water", "k_road",
                          "quality_factor"]].iterrows():
        print(
            f"  {idx:>7}  {row['service_level']:<10}  "
            f"{row['area_m2']/1e4:>11.2f}  "
            f"{row['k_zone']:>7.3f}  {row['k_ndvi']:>7.3f}  "
            f"{row['k_water']:>8.3f}  {row['k_road']:>7.3f}  "
            f"{row['quality_factor']:>7.3f}"
        )
    print()


def _pick_file(data_dir: str, title: str) -> pathlib.Path | None:
    files = sorted(pathlib.Path(data_dir).glob("*.gpkg"))
    if not files:
        print(f"  В {data_dir}/ нет .gpkg-файлов.")
        return None
    print(f"\n{title}")
    for i, f in enumerate(files, start=1):
        size_kb = f.stat().st_size // 1024
        print(f"  {i:>2})  {f.name:<40}  {size_kb:>6} КБ")
    idx = _ask(f"Номер файла [1–{len(files)}]: ", cast=int) - 1
    if not (0 <= idx < len(files)):
        print("  ✗ Номер вне диапазона.")
        return None
    return files[idx]


def _print_comparison(m0: dict, m1: dict) -> None:
    rows = [
        ("A_h mean",   "A_h_mean",      ".3f"),
        ("A_h median", "A_h_median",    ".3f"),
        ("CV",         "CV",            ".3f"),
        ("Дефицит",    "deficit_share", ".1%"),
        ("S_A",        "S_A",           ".3f"),
    ]
    print(f"\n  {'Метрика':<15}  {'Baseline':>10}  {'Сценарий':>10}  {'Δ':>10}")
    print("  " + "─" * 52)
    for label, key, fmt in rows:
        v0, v1 = m0[key], m1[key]
        print(f"  {label:<15}  {format(v0, fmt):>10}  "
              f"{format(v1, fmt):>10}  {v1 - v0:>+10.3f}")


# -------------------------------------------------------------------- #
#                              ТОЧКА ВХОДА                             #
# -------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Сценарный анализ")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--territory", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    # CRS автоопределение
    if not cfg.get("crs"):
        _bnd = gpd.read_file(cfg["paths"]["boundary"], rows=1)
        _crs = _bnd.crs
        if _crs is None:
            raise ValueError("boundary.gpkg не содержит CRS. Задайте crs в config.yaml.")
        if _crs.is_geographic:
            _c = _bnd.to_crs(4326).geometry.centroid.iloc[0]
            _zone = int((_c.x + 180) / 6) + 1
            cfg["crs"] = f"EPSG:{(32600 if _c.y >= 0 else 32700) + _zone}"
        else:
            cfg["crs"] = str(_crs)

    # ── Baseline ────────────────────────────────────────────────────────────
    print("\n══ Подготовка данных ══════════════════════════════════════")
    boundary, meta = select_territory(
        cfg["paths"]["boundary"], cfg["crs"],
        territory_name=args.territory,
    )
    out_dir = pathlib.Path(cfg["paths"]["outputs_dir"]) / _safe_dirname(meta["name"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data   = load_all(cfg, boundary)
    _network_type = cfg.get("network_type", "walk")
    _network_file = cfg["paths"].get("network_file", "") or None
    G      = load_or_download_graph(
        boundary, cfg["crs"],
        network_type=_network_type,
        network_file=_network_file,
    )
    _hw_file = cfg["paths"].get("highway_file", "") or None
    hw_osm = load_motorway_edges(boundary, cfg["crs"], highway_file=_hw_file)
    rec_q  = apply_quality(data["recreation"], data["zoning"],
                           data["water"], hw_osm, data["ndvi_path"], cfg)

    b_nodes = build_node_index(G, data["buildings"], "centroid")
    r_nodes = build_node_index(G, rec_q, "centroid")
    max_r   = max(s["radius"] for s in cfg["service_levels"].values())
    dist    = build_distance_matrix(G, b_nodes, r_nodes, max_r)

    bld0, rec0, m0, thr = run_baseline(data["buildings"], rec_q, dist, cfg)

    print(f"\n  Baseline · {meta['name']}")
    print(f"  A_h={m0['A_h_mean']:.3f}  дефицит={m0['deficit_share']:.1%}  S_A={m0['S_A']:.3f}")



    # ── Меню ────────────────────────────────────────────────────────────────
    print("\n══ Тип сценария ═══════════════════════════════════════════")
    print("  1)  Новый рекреационный объект  (файл из data/)")
    print("  2)  Новое жилое здание          (файл из data/)")
    print("  3)  Улучшить k_zone / k_ndvi существующего объекта")
    print("  0)  Выход")

    choice = _ask("\nВаш выбор [0–3]: ", valid=["0", "1", "2", "3"])
    if choice == "0":
        return

    new_recr      = None
    new_buildings = None
    modify        = None
    scenario_name = "scenario"

    # ── 1: новый рекреационный объект ───────────────────────────────────────
    if choice == "1":
        # 1а. Карта качества с индексами
        print("\n── Карта качества объектов ──")
        _map_quality_indexed(
            rec0, data["boundary"],
            out_dir / "_quality_indexed.png",
            title=f"Качество объектов · {meta['name']}",
        )

        # 1б. Выбрать файл
        fp = _pick_file("data", "Файлы в data/ — выберите слой новых объектов:")
        if fp is None:
            return
        new_recr = gpd.read_file(str(fp))
        scenario_name = f"infra_{fp.stem}"
        print(f"  Загружено {len(new_recr)} объектов из «{fp.name}»")

        # 1в. Превью: новые объекты на карте (открывается автоматически)
        print("\n── Превью расположения новых объектов ──")
        _map_preview_new_recr(
            rec0, new_recr, data["boundary"],
            out_dir / "_preview_new_recr.png",
            show=True,
        )



    # ── 2: новое жилое здание ───────────────────────────────────────────────
    elif choice == "2":
        fp = _pick_file("data", "Файлы в data/ — выберите слой новых зданий:")
        if fp is None:
            return
        new_buildings = gpd.read_file(str(fp))
        scenario_name = f"demo_{fp.stem}"
        print(f"  Загружено {len(new_buildings)} зданий из «{fp.name}»")

    # ── 3: улучшить существующий объект ─────────────────────────────────────
    elif choice == "3":
        # 3а. Карта качества с индексами
        print("\n── Карта качества объектов ──")
        _map_quality_indexed(
            rec0, data["boundary"],
            out_dir / "_quality_indexed.png",
            title=f"Качество объектов · {meta['name']}",
        )

        # 3б. Таблица объектов
        _print_recr_table(rec0)

        # 3в. Выбор объекта
        obj_idx = _ask("Введите индекс объекта: ", cast=int)
        if obj_idx not in rec0.index:
            print(f"  ✗ Индекс {obj_idx} не найден.")
            return

        # 3г. Превью: выбранный объект подсвечен (открывается автоматически)
        print("\n── Превью выбранного объекта ──")
        _map_preview_highlight(
            rec0, obj_idx, data["boundary"],
            out_dir / f"_preview_highlight_{obj_idx}.png",
            show=True,
        )

        # 3д. Ввод новых коэффициентов
        cur_kz = float(rec0.at[obj_idx, "k_zone"])
        cur_kn = float(rec0.at[obj_idx, "k_ndvi"])

        print(f"\n  Текущий k_zone = {cur_kz:.3f}  (0.60–1.20, Enter = без изменений)")
        k_zone_new = _ask("  Новый k_zone: ", default=str(cur_kz), cast=float)
        print(f"  Текущий k_ndvi = {cur_kn:.3f}  (0.50–1.30, Enter = без изменений)")
        k_ndvi_new = _ask("  Новый k_ndvi: ", default=str(cur_kn), cast=float)

        modify = {obj_idx: {"k_zone": k_zone_new, "k_ndvi": k_ndvi_new}}
        scenario_name = f"infra_modify_{obj_idx}"
        print(f"  → modify = {modify}")

        # 3е. Подтверждение
        ok = _ask("\nЗапустить расчёт сценария? [y/n]: ",
                  valid=["y", "n", "Y", "N", "д", "н"])
        if ok.lower() in ("n", "н"):
            print("  Отменено.")
            return

    # ── Расчёт ──────────────────────────────────────────────────────────────
    print("\n══ Расчёт ═════════════════════════════════════════════════")

    if choice in ("1", "3"):
        bld1, rec1, m1 = run_infrastructure_scenario(
            buildings=data["buildings"], recreation=rec0,
            dist_matrix=dist, G=G, building_nodes=b_nodes,
            cfg=cfg, baseline_thresholds=thr,
            new_recr=new_recr,
            modify=modify,
            zoning=data.get("zoning"),
            water=data.get("water"),
            highway_osm=hw_osm,
            ndvi_path=data.get("ndvi_path"),
        )
    else:
        bld1, rec1, m1 = run_demographic_scenario(
            buildings=data["buildings"], recreation=rec0,
            dist_matrix=dist,
            cfg=cfg, baseline_thresholds=thr,
            new_buildings=new_buildings,
        )

    # ── Результаты ──────────────────────────────────────────────────────────
    print(f"\n══ Результаты: {scenario_name} ═══════════════════════════")
    _print_comparison(m0, m1)

    export_results(bld0, rec0, m0, out_dir, scenario_name="baseline")
    export_results(bld1, rec1, m1, out_dir, scenario_name=scenario_name)
    export_comparison({"baseline": m0, scenario_name: m1},
                      out_dir, name=f"cmp_{scenario_name}")

    plot_deficit_map(bld1,
                     out_dir / f"map_deficit_{scenario_name}.png",
                     title=f"Дефицит · {scenario_name} · {meta['name']}",
                     boundary=data["boundary"])
    plot_accessibility_map(bld1, "A_h",
                           out_dir / f"map_ah_{scenario_name}.png",
                           title=f"A_h · {scenario_name} · {meta['name']}")

    # Для сценария 1 — итоговая карта с подсвеченными новыми объектами
    if choice == "1" and new_recr is not None:
        print("\n── Итоговая карта: новые объекты на фоне результата ──")
        _map_preview_new_recr(
            rec1, new_recr, data["boundary"],
            out_dir / f"map_result_new_recr_{scenario_name}.png",
        )

    print(f"\n  Файлы сохранены: {out_dir}")


if __name__ == "__main__":
    main()
