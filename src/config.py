"""Загрузка и валидация config.yaml."""
from __future__ import annotations
import logging
import pathlib
import yaml


def load_config(path: str | pathlib.Path = "config.yaml") -> dict:
    """Прочитать YAML и вернуть словарь с параметрами расчёта."""
    p = pathlib.Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"config не найден: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(p)
    return cfg


def setup_logging(cfg: dict) -> None:
    """Настроить корневой логгер по ключу ``verbosity`` из конфига.

    Поддерживаемые уровни: DEBUG, INFO, WARNING, ERROR.
    По умолчанию — INFO.  Устанавливается один раз при старте run.py;
    при повторном вызове перезаписывает предыдущий уровень (force=True).
    """
    level_name: str = str(cfg.get("verbosity", "INFO")).upper()
    level: int = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(levelname)-8s  %(name)s: %(message)s",
        force=True,
    )
