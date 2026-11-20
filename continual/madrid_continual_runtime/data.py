"""Metadata reader shared by the unchanged PECPM adapter; no few-shot plans."""
import json
import numpy as np


def cache_metadata(inputs):
    value=inputs['__metadata__']
    if isinstance(value,np.ndarray): value=value.item()
    if isinstance(value,bytes): value=value.decode('utf-8')
    return json.loads(value) if isinstance(value,str) else dict(value)
