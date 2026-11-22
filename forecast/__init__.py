"""Conventional forecasters used in the full-data comparison."""

from importlib import import_module


MODELS = ("TQNET", "GWNET", "STID", "PATCHTST")


def canonical_model(name):
    name = name.strip().upper().replace("-", "").replace("_", "")
    if name in ("GRAPHWAVENET", "GWN"):
        name = "GWNET"
    if name not in MODELS:
        raise ValueError(f"Unknown conventional baseline {name!r}; choose {', '.join(MODELS)}")
    return name


def implementation(name):
    return import_module(f".{canonical_model(name).lower()}", __name__)
