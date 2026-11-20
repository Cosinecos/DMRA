"""The four existing continual methods, isolated for full-data Madrid runs."""
from importlib import import_module

MODELS=('EAC','PECPM','STKEC','TS')
MODULES={'EAC':'eac','PECPM':'pecpm','STKEC':'stkec','TS':'trafficstream'}
ALIASES={'SKTEC':'STKEC','TRAFFICSTREAM':'TS','TRAFFIC_STREAM':'TS'}


def canonical_model(value):
    name=ALIASES.get(str(value).upper().replace('-','_'),str(value).upper().replace('-','_'))
    if name not in MODELS: raise ValueError(f'Choose a Madrid continual model from {MODELS}')
    return name


def implementation(value):
    return import_module('madrid_continual_runtime.baselines.'+MODULES[canonical_model(value)]+'.adapter')
