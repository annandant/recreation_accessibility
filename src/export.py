"""Сохранение результатов: GPKG, CSV/XLSX со сводной статистикой."""
from __future__ import annotations
import logging
import pathlib

import pandas as pd
import geopandas as gpd

log = logging.getLogger(__name__)


def _drop_centroid(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return g.drop(columns="centroid", errors="ignore")


def export_results(
    buildings: gpd.GeoDataFrame,
    recreation: gpd.GeoDataFrame,
    metrics: dict,
    out_dir: str | pathlib.Path,
    scenario_name: str = "baseline",
) -> dict:
    """Сохранить buildings/recreation в GPKG и сводку в CSV/XLSX."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bld_path = out / f"buildings_{scenario_name}.gpkg"
    rec_path = out / f"recreation_{scenario_name}.gpkg"
    _drop_centroid(buildings).to_file(bld_path, driver="GPKG")
    # rec_idx — числовой индекс GeoDataFrame; нужен для сценарного анализа
    # (выбор объекта по номеру в run_scenario.py и просмотр в ГИС).
    _rec = _drop_centroid(recreation).copy()
    _rec["rec_idx"] = _rec.index
    _rec.to_file(rec_path, driver="GPKG")
    log.info("Сохранено: %s, %s", bld_path.name, rec_path.name)

    # сводная статистика
    df = pd.DataFrame([metrics])
    csv_path = out / f"summary_{scenario_name}.csv"
    xlsx_path: pathlib.Path | None = out / f"summary_{scenario_name}.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as e:
        log.warning("XLSX не сохранён: %s", e)
        xlsx_path = None

    return {
        "buildings_gpkg": str(bld_path),
        "recreation_gpkg": str(rec_path),
        "summary_csv": str(csv_path),
        "summary_xlsx": str(xlsx_path) if xlsx_path else None,
    }


def export_comparison(
    metrics_by_scenario: dict[str, dict],
    out_dir: str | pathlib.Path,
    name: str = "comparison",
) -> dict:
    """Сравнительная таблица метрик для всех сценариев."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics_by_scenario).T
    df.index.name = "scenario"
    csv = out / f"{name}.csv"
    xlsx: pathlib.Path | None = out / f"{name}.xlsx"
    df.to_csv(csv, encoding="utf-8-sig")
    try:
        df.to_excel(xlsx)
    except Exception as e:
        log.warning("XLSX сравнения не сохранён: %s", e)
        xlsx = None
    return {"csv": str(csv), "xlsx": str(xlsx) if xlsx else None}
