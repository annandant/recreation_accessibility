"""Каскадный выбор территории по атрибутам boundary.gpkg.

Иерархия в boundary.gpkg:
    ADMIN_LVL=4  → город (Санкт-Петербург)
    ADMIN_LVL=5  → административный район
    ADMIN_LVL=8  → муниципальное образование / посёлок

Алгоритм:
    1) пользователю показывается список всех NAME для самого верхнего уровня;
    2) после выбора показываются только те записи следующего уровня,
       чей представительный центроид лежит ВНУТРИ выбранного полигона
       (пространственный фильтр — устойчив к отсутствию/несовпадению
       атрибутов ADMIN_L4 / ADMIN_L5);
    3) на любом шаге можно «остановиться здесь» — текущий полигон
       становится границей расчёта.
"""
from __future__ import annotations
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon


def _prompt_choice(options: list[str], header: str, allow_stop: bool) -> Optional[int]:
    """Показать пронумерованный список и получить выбор пользователя.

    Возвращает индекс выбранной опции; None — если пользователь решил
    остановиться на текущем уровне.
    """
    print(f"\n=== {header} ===")
    for i, name in enumerate(options, start=1):
        print(f"  {i:>3}. {name}")
    if allow_stop:
        print(f"  {0:>3}. ← остановиться на текущем уровне")
    while True:
        raw = input("Выбор (число или часть названия): ").strip()
        if not raw:
            continue
        if raw.lstrip("-").isdigit():
            idx = int(raw)
            if allow_stop and idx == 0:
                return None
            if 1 <= idx <= len(options):
                return idx - 1
            print("  Неверный номер, попробуйте ещё раз.")
            continue
        matches = [i for i, n in enumerate(options) if raw.lower() in n.lower()]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            print("  Совпадений не найдено.")
        else:
            print(f"  Совпадений несколько: {[options[i] for i in matches][:5]}…")


def _spatial_children(
    boundary: gpd.GeoDataFrame,
    level: int | str,
    parent_geom,
) -> gpd.GeoDataFrame:
    """Вернуть записи ``level``, чьи геометрии пересекают ``parent_geom``.

    Используется ``representative_point()`` (гарантированно внутри полигона),
    что устойчиво к вложенным/частично перекрывающимся границам.
    """
    df = boundary[boundary["ADMIN_LVL"] == str(level)].copy()
    if df.empty:
        return df
    # Приводим parent_geom к тому же CRS, что и boundary
    mask = df.geometry.representative_point().within(parent_geom)
    # fallback: если ни один не попал внутрь — попробуем intersects (граничные случаи)
    if mask.sum() == 0:
        mask = df.geometry.intersects(parent_geom)
    return df[mask].sort_values("NAME").reset_index(drop=True)


def select_territory(
    boundary_path: str,
    target_crs: str,
    territory_name: Optional[str] = None,
) -> tuple[gpd.GeoDataFrame, dict]:
    """Каскадный выбор территории.

    Параметры
    ---------
    boundary_path : путь к boundary.gpkg
    target_crs    : целевая СК (для приведения геометрии)
    territory_name: если задано — берётся напрямую (без интерактива).

    Возвращает
    ----------
    (boundary_gdf, meta) — GeoDataFrame с выбранным полигоном и метаданные:
        meta = {"name": str, "admin_lvl": int, "osm_id": int}
    """
    boundary = gpd.read_file(boundary_path).to_crs(target_crs)

    # --- неинтерактивный путь (по имени) ------------------------------
    if territory_name is not None:
        sel = boundary[
            boundary["NAME"].fillna("").str.casefold() == territory_name.casefold()
        ]
        if sel.empty:
            sel = boundary[
                boundary["NAME"].fillna("").str.contains(
                    territory_name, case=False, na=False
                )
            ]
        if sel.empty:
            raise ValueError(f"Территория '{territory_name}' не найдена в boundary.gpkg")
        if len(sel) > 1:
            # предпочесть наибольший уровень детализации (max ADMIN_LVL)
            best_lvl = sel["ADMIN_LVL"].astype(str).map(lambda v: int(v) if v.isdigit() else 0).max()
            sel = sel[sel["ADMIN_LVL"].astype(str).map(lambda v: int(v) if v.isdigit() else 0) == best_lvl]
            if len(sel) > 1:
                raise ValueError(
                    f"Найдено {len(sel)} совпадений по '{territory_name}': "
                    f"{sel['NAME'].tolist()[:5]} — уточните название."
                )
        row = sel.iloc[0]
        return _finalize(sel, row)

    # --- интерактивный каскад ----------------------------------------
    levels = sorted(
        boundary["ADMIN_LVL"].dropna().unique().tolist(),
        key=lambda v: int(v) if str(v).isdigit() else 0,
    )
    print(f"[territory] Доступные уровни ADMIN_LVL: {levels}")

    chosen_row: Optional[pd.Series] = None
    chosen_geom = None          # объединённая геометрия выбранной территории

    for li, lvl in enumerate(levels):
        if chosen_geom is None:
            # первый уровень — показываем всё
            df = boundary[boundary["ADMIN_LVL"] == str(lvl)].sort_values("NAME").reset_index(drop=True)
        else:
            # последующие уровни — только те, что лежат внутри выбранного полигона
            df = _spatial_children(boundary, lvl, chosen_geom)

        if df.empty:
            print(f"[territory] На уровне {lvl} нет записей для выбранного родителя.")
            break

        names = df["NAME"].fillna("(без названия)").tolist()
        allow_stop = chosen_row is not None
        idx = _prompt_choice(
            names,
            header=f"Уровень ADMIN_LVL = {lvl} ({len(names)} объектов)",
            allow_stop=allow_stop,
        )
        if idx is None:
            break

        chosen_row = df.iloc[idx]
        chosen_geom = chosen_row.geometry
        print(f"  → выбрано: {chosen_row['NAME']} (OSM_ID={chosen_row['OSM_ID']})")

        if li == len(levels) - 1:
            break

    if chosen_row is None:
        raise RuntimeError("Территория не выбрана.")

    sel = boundary[boundary["OSM_ID"] == int(chosen_row["OSM_ID"])]
    return _finalize(sel, chosen_row)


def _finalize(sel: gpd.GeoDataFrame, row: pd.Series) -> tuple[gpd.GeoDataFrame, dict]:
    meta = {
        "name": str(row["NAME"]),
        "admin_lvl": int(row["ADMIN_LVL"]) if str(row["ADMIN_LVL"]).isdigit() else None,
        "osm_id": int(row["OSM_ID"]) if pd.notna(row["OSM_ID"]) else None,
    }
    print(f"[territory] Итоговая территория: {meta}")
    return sel.copy(), meta