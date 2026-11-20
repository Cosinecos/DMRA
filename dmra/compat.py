"""Translate historical experiment identifiers when loading original checkpoints."""

_NAMES = {
    "integrated_shared_kv": "dmra",
    "PSRAG_IntegratedSharedKV_Model": "DMRA",
    "MadridIntegratedSharedKVPSRAG": "MadridDMRA",
    "PSRAG_REFINE_V6_STANDARD_GCN_DYN_FRETS_LORA_TREND_QKNORM_WEEK_SHARED_KV_GCN_ADAPTER_LORA": "DMRA",
    "PSRAG_REFINE_V6_STANDARD_GCN_DYN_FRETS_LORA_TREND_QKNORM_WEEK_SHARED_KV_GCN_ADAPTER_LORA_MADRID_MMD": "DMRA",
}


def canonical_metadata(metadata):
    if isinstance(metadata, dict):
        return {key: canonical_metadata(value) for key, value in metadata.items()}
    if isinstance(metadata, str):
        return _NAMES.get(metadata, metadata)
    return metadata
