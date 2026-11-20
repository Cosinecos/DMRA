"""Continual forecasting baseline registry."""

from importlib import import_module

MODELS = ("STBP", "STRAP", "EAC", "PECPM", "STKEC", "TS")
MODULES = {"STBP": "stbp", "STRAP": "strap", "EAC": "eac",
           "PECPM": "pecpm", "STKEC": "stkec", "TS": "trafficstream"}
ALIASES = {"SKTEC": "STKEC", "TRAFFICSTREAM": "TS", "TRAFFIC_STREAM": "TS"}


def canonical_model(name):
    value = name.upper().replace("-", "_")
    value = ALIASES.get(value, value)
    if value not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose {MODELS}")
    return value


def implementation(name):
    return import_module(f"baselines.{MODULES[canonical_model(name)]}.adapter")
