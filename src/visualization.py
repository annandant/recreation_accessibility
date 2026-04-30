"""Картографическая визуализация результатов"""
from __future__ import annotations
import logging
import pathlib
from typing import Optional

import contextily as ctx
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

log = logging.getLogger(__name__)

# -------------------------------------------------------------------- #
# Палитры
# -------------------------------------------------------------------- #
CMAP_CALC = LinearSegmentedColormap.from_list(
    "calc_blue_purple", ["#88addc", "#6b3ca8", "#9f03a6"], N=256
)
# ИНВЕРТИРОВАНО: высокое quality_factor → жёлто-зелёный; низкое → синий
CMAP_QUALITY = LinearSegmentedColormap.from_list(
    "quality_blue_lime", ["#2c7fb8", "#88c4d6", "#c3ef3d"], N=256
)

DEFICIT_COLORS: dict[str, str] = {
    "deficit": "#9f03a6",
    "normal":  "#6b3ca8",
    "surplus": "#88addc",
}
DEFICIT_LABELS_RU: dict[str, str] = {
    "deficit": "Дефицит",
    "normal":  "Норма",
    "surplus": "Профицит",
}

_FIELD_LABELS: dict[str, str] = {
    "A_400":  "Местная доступность A₄₀₀ (400 м)",
    "A_1500": "Районная доступность A₁₅₀₀ (1 500 м)",
    "A_6000": "Городская доступность A₆₀₀₀ (6 000 м)",
    "A_h":    "Интегральная доступность A_h",
}


# -------------------------------------------------------------------- #
# Вспомогательные функции
# -------------------------------------------------------------------- #

def _basemap(ax, crs) -> None:
    """Добавление подложки"""
    try:
        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.PositronNoLabels,
            crs=str(crs),
            zoom="auto",
        )
    except Exception as e:
        log.warning("Подложка картографии недоступна: %s", e)


def _to_wm(gdf: Optional[gpd.GeoDataFrame]) -> Optional[gpd.GeoDataFrame]:
    return gdf.to_crs(epsg=3857) if gdf is not None else None


def _save(fig, path: str | pathlib.Path) -> None:
    """Сохранить фигуру. bbox_inches='tight' включает внешние легенды."""
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Карта сохранена: %s", path)


def _outside_colorbar(
    fig,
    ax,
    cmap,
    vmin: float,
    vmax: float,
    label: str,
    n_ticks: int = 5,
) -> mpl.colorbar.Colorbar:
    """Создать colorbar справа от ax, за пределами карты.

    Использует fig.colorbar() с fraction/pad — совместимо с contextily.
    """
    sm = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, shrink=0.80, aspect=28)
    cb.set_label(label, fontsize=10, labelpad=10)
    cb.ax.tick_params(labelsize=9)
    # Аккуратные метки: n_ticks равномерно от vmin до vmax
    ticks = np.linspace(vmin, vmax, n_ticks)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.3g}" for t in ticks])
    return cb


def _voronoi_zones(
    buildings: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
) -> Optional[gpd.GeoDataFrame]:
    """Зоны из центроидов зданий, обрезанные по boundary."""
    from shapely.ops import voronoi_diagram
    from shapely.geometry import MultiPoint

    try:
        boundary_union = boundary.unary_union
        centroids = buildings.geometry.centroid
        pts = MultiPoint(list(centroids))
        envelope = boundary_union.buffer(500)
        regions = voronoi_diagram(pts, envelope=envelope)
        vor_gdf = gpd.GeoDataFrame(geometry=list(regions.geoms), crs=buildings.crs)
        vor_clipped = gpd.clip(vor_gdf, boundary_union)
        if vor_clipped.empty:
            return None
        centroid_gdf = gpd.GeoDataFrame(
            {"access_class": buildings["access_class"].values},
            geometry=centroids.values,
            crs=buildings.crs,
        )
        result = gpd.sjoin_nearest(
            vor_clipped[["geometry"]],
            centroid_gdf[["geometry", "access_class"]],
        )
        return result[~result.index.duplicated(keep="first")]
    except Exception as e:
        log.warning("зонирование не удалось: %s — используем точки зданий", e)
        return None


# -------------------------------------------------------------------- #
#                        ОСНОВНЫЕ КАРТЫ                                #
# -------------------------------------------------------------------- #

def plot_quality_map(
    recreation: gpd.GeoDataFrame,
    output_path: str | pathlib.Path,
    title: Optional[str] = None,
) -> None:
    """Качество рекреационных объектов"""
    rec = _to_wm(recreation)
    vmin = float(rec["quality_factor"].min())
    vmax = float(rec["quality_factor"].max())

    fig, ax = plt.subplots(figsize=(14, 9))
    rec.plot(
        column="quality_factor",
        cmap=CMAP_QUALITY,
        vmin=vmin, vmax=vmax,
        legend=False,
        markersize=40,
        alpha=0.9,
        ax=ax,
        zorder=3,
    )
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    _basemap(ax, rec.crs)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(title or "Качество рекреационных объектов", pad=8)
    ax.set_axis_off()

    _outside_colorbar(fig, ax, CMAP_QUALITY, vmin, vmax, "Коэффициент качества")
    _save(fig, output_path)


def plot_accessibility_map(
    buildings: gpd.GeoDataFrame,
    field: str,
    output_path: str | pathlib.Path,
    title: Optional[str] = None,
) -> None:
    """Карта обеспеченности (A_400 / A_1500 / A_6000 / A_h)"""
    bld = _to_wm(buildings)
    vmin = float(bld[field].min())
    vmax = float(bld[field].max())
    legend_label = _FIELD_LABELS.get(field, field)

    fig, ax = plt.subplots(figsize=(14, 9))
    bld.plot(
        column=field,
        cmap=CMAP_CALC,
        vmin=vmin, vmax=vmax,
        legend=False,
        markersize=10,
        alpha=0.9,
        ax=ax,
        zorder=3,
    )
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    _basemap(ax, bld.crs)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(title or f"Обеспеченность · {field}", pad=8)
    ax.set_axis_off()

    _outside_colorbar(fig, ax, CMAP_CALC, vmin, vmax, legend_label)
    _save(fig, output_path)


def plot_deficit_map(
    buildings: gpd.GeoDataFrame,
    output_path: str | pathlib.Path,
    title: Optional[str] = None,
    boundary: Optional[gpd.GeoDataFrame] = None,
) -> None:
    """Карта зон дефицита/нормы/профицита"""
    bld = _to_wm(buildings)
    fig, ax = plt.subplots(figsize=(14, 9))

    # --- территориальная заливка (Voronoi) ---
    if boundary is not None:
        bnd_wm = _to_wm(boundary)
        zones = _voronoi_zones(bld, bnd_wm)
    else:
        zones = None

    if zones is not None:
        for cls, color in DEFICIT_COLORS.items():
            sub = zones[zones["access_class"] == cls]
            if not sub.empty:
                sub.plot(ax=ax, color=color, alpha=0.78, linewidth=0, zorder=3)
        bnd_wm.boundary.plot(ax=ax, color="#444444", linewidth=0.9, zorder=4)
    else:
        for cls, color in DEFICIT_COLORS.items():
            sub = bld[bld["access_class"] == cls]
            if not sub.empty:
                sub.plot(ax=ax, color=color, markersize=14, alpha=0.9, zorder=3)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    _basemap(ax, bld.crs)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(title or "Зоны дефицита обеспеченности", pad=8)
    ax.set_axis_off()

    # Легенда справа от карты
    handles = [
        mpatches.Patch(color=DEFICIT_COLORS[k], label=DEFICIT_LABELS_RU[k])
        for k in DEFICIT_COLORS
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        title="Зона обеспеченности",
        title_fontsize=10,
        fontsize=10,
        framealpha=0.95,
        edgecolor="#cccccc",
    )
    _save(fig, output_path)


def plot_stress_map(
    buildings: gpd.GeoDataFrame,
    output_path: str | pathlib.Path,
    title: Optional[str] = None,
) -> None:
    """Карта индивидуального стресса S_i"""
    bld = _to_wm(buildings)
    fig, ax = plt.subplots(figsize=(14, 9))
    bld.plot(
        column="stress_i",
        cmap=CMAP_CALC,
        vmin=0, vmax=1,
        legend=False,
        markersize=12,
        alpha=0.9,
        ax=ax,
        zorder=3,
    )
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    _basemap(ax, bld.crs)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(title or "Рекреационный стресс S_i", pad=8)
    ax.set_axis_off()

    _outside_colorbar(fig, ax, CMAP_CALC, 0.0, 1.0,
                      "Стресс-индекс S_i\n[0 — нет дефицита; 1 — максимум]",
                      n_ticks=6)
    _save(fig, output_path)


# =========================================================================
# === DEV_VISUALS ============================================ НАЧАЛО ====
# =========================================================================
def plot_graph(
    G,
    buildings: Optional[gpd.GeoDataFrame] = None,
    recreation: Optional[gpd.GeoDataFrame] = None,
    output_path: str | pathlib.Path = "outputs/graph.png",
    title: Optional[str] = None,
    boundary: Optional[gpd.GeoDataFrame] = None,
) -> None:
    """Граф пешеходной сети (обрезан по boundary/buildings, подложка EPSG:3857)."""
    import osmnx as ox

    # Конвертируем граф в GeoDataFrame → EPSG:3857, чтобы избежать проблем
    # с форматом CRS-строки при передаче в contextily.
    graph_crs = G.graph.get("crs", None)
    try:
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
        if graph_crs and nodes_gdf.crs is None:
            nodes_gdf = nodes_gdf.set_crs(graph_crs, allow_override=True)
            edges_gdf = edges_gdf.set_crs(graph_crs, allow_override=True)
        nodes_wm = nodes_gdf.to_crs(epsg=3857)
        edges_wm = edges_gdf.to_crs(epsg=3857)
        _use_gdfs = True
    except Exception as e:
        log.warning("graph_to_gdfs не удался: %s — использую ox.plot_graph", e)
        _use_gdfs = False

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor("white")

    if _use_gdfs:
        edges_wm.plot(ax=ax, color="#6b3ca8", linewidth=0.4, zorder=2, alpha=0.6)
        nodes_wm.plot(ax=ax, color="#88addc", markersize=2, zorder=3)
    else:
        # Крайний случай: ox.plot_graph рисует в текущий ax
        ox.plot_graph(G, ax=ax, show=False, close=False, bgcolor="none",
                      node_color="#88addc", node_size=2,
                      edge_color="#6b3ca8", edge_linewidth=0.4)

    # Слои данных
    if buildings is not None and not buildings.empty:
        bld_wm = buildings.to_crs(epsg=3857)
        c = bld_wm.geometry.centroid
        ax.scatter([p.x for p in c], [p.y for p in c],
                   s=8, c="#2c7fb8", alpha=0.7, zorder=4, label="Жилые дома")
    if recreation is not None and not recreation.empty:
        rec_wm = recreation.to_crs(epsg=3857)
        if "centroid" in recreation.columns:
            cent_wm = gpd.GeoDataFrame(
                geometry=list(recreation["centroid"]), crs=recreation.crs
            ).to_crs(epsg=3857).geometry
        else:
            cent_wm = rec_wm.geometry.centroid
        ax.scatter([p.x for p in cent_wm], [p.y for p in cent_wm],
                   s=40, c="#c3ef3d", alpha=0.95, zorder=5,
                   marker="^", edgecolors="#2c7fb8", linewidths=0.5,
                   label="Рекреационные объекты")

    # --- Обрезаем вид по зданиям, а не по всему буферу графа ----------------
    # Здания — наиболее надёжный якорь: они точно расположены на территории.
    # Boundary может иметь вытянутую форму (дамба, административный выступ)
    # и давать bbox, выходящий на материк.
    _viewport_ref = buildings if (buildings is not None and not buildings.empty) \
                    else boundary
    if _viewport_ref is not None and not _viewport_ref.empty:
        _vp_wm = _viewport_ref.to_crs(epsg=3857)
        mnx, mny, mxx, mxy = _vp_wm.total_bounds
        mg = max((mxx - mnx), (mxy - mny)) * 0.06   # 6% отступ
        ax.set_xlim(mnx - mg, mxx + mg)
        ax.set_ylim(mny - mg, mxy + mg)
    if boundary is not None and not boundary.empty:
        boundary.to_crs(epsg=3857).boundary.plot(
            ax=ax, color="#444444", linewidth=1.0, zorder=6)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    _basemap(ax, "EPSG:3857")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(title or "Пешеходная сеть", pad=8)
    ax.set_axis_off()

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=10,
    )
    _save(fig, output_path)


def plot_quality_metrics(
    recreation: gpd.GeoDataFrame,
    output_path: str | pathlib.Path = "outputs/quality_metrics.png",
    title: Optional[str] = None,
) -> None:
    """диагностика качества рекреационных объектов"""
    rec = recreation.copy().reset_index(drop=True)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, :])
    norm = mpl.colors.Normalize(vmin=rec["quality_factor"].min(),
                                vmax=rec["quality_factor"].max())
    colors = CMAP_QUALITY(norm(rec["quality_factor"]))
    ax1.bar(range(len(rec)), rec["quality_factor"], color=colors)
    ax1.set_title("Коэффициент качества по объектам")
    ax1.set_xlabel("Объект")
    ax1.set_ylabel("Коэффициент качества")

    ax2 = fig.add_subplot(gs[1, 0])
    bp = ax2.boxplot(
        [rec["k_zone"], rec["k_ndvi"], rec["k_water"], rec["k_road"]],
        labels=["k_zone", "k_ndvi", "k_water", "k_road"],
        patch_artist=True,
    )
    for patch, c in zip(bp["boxes"], ["#c3ef3d", "#88c4d6", "#88addc", "#2c7fb8"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    ax2.set_title("Распределение коэффициентов качества")
    ax2.set_ylabel("Значение коэффициента")

    ax3 = fig.add_subplot(gs[1, 1])
    sc = ax3.scatter(rec["capacity"], rec["capacity_adj"],
                     c=rec["quality_factor"], cmap=CMAP_QUALITY, s=30, alpha=0.85)
    cb = fig.colorbar(sc, ax=ax3, fraction=0.04, pad=0.02, shrink=0.85)
    cb.set_label("Коэффициент качества", fontsize=9)
    lim = max(rec["capacity"].max(), rec["capacity_adj"].max())
    ax3.plot([0, lim], [0, lim], "--", color="#6b3ca8", alpha=0.5)
    ax3.set_xlabel("Нормативная вместимость, чел.")
    ax3.set_ylabel("Скорректированная вместимость C'_j, чел.")
    ax3.set_title("Эффект корректировки качества")

    fig.suptitle(title or "Метрики качества рекреационных объектов")
    fig.tight_layout()
    _save(fig, output_path)
# =========================================================================
# === DEV_VISUALS ============================================ КОНЕЦ ====
# =========================================================================