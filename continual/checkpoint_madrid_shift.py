"""Shared utilities for the Madrid continual forecasting protocol."""

PROTOCOL_VERSION = 1

def protocol(args):
    return dict(version=PROTOCOL_VERSION, dataset="MADRID_2022_2026_MMD",
                data_manifest_sha256=str(args.data_manifest_sha256),
                selection_result_sha256=str(args.selection_result_sha256),
                reference_anchor_sha256=str(args.reference_anchor_sha256),
                calendar="actual_madrid_selected_month_wall_clock_15min",
                missing_labels="explicit_mask_zero_is_valid_and_DST_window_exclusion",
                validation="global_observed_MAE", unit="vehicles_per_hour",
                mape="observed_positive_targets_only")
