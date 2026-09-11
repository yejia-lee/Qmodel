"""Shared evaluation protocol for the MDS-ED Q-model experiments."""

import numpy as np


BASE_TRAIN_FOLDS = tuple(range(1, 16))
QMODEL_FOLDS = tuple(range(16, 19))
VALIDATION_FOLD = 19
TEST_FOLD = 20
SENSITIVITY_TARGET = 0.80

# Backward-compatible alias for code that uses this name for preprocessing
# medians. Base predictors must use BASE_TRAIN_FOLDS explicitly.
TRAIN_FOLDS = BASE_TRAIN_FOLDS


def add_protocol_fold(df, source_column="general_strat_fold"):
    """Return a copy with folds represented as the paper's 1-based labels."""
    result = df.copy()
    folds = result[source_column].astype(int)
    if folds.min() == 0 and folds.max() <= 19:
        folds = folds + 1
    result["protocol_fold"] = folds
    expected = set(BASE_TRAIN_FOLDS) | set(QMODEL_FOLDS) | {VALIDATION_FOLD, TEST_FOLD}
    observed = set(folds.unique())
    if not observed.issubset(expected):
        raise ValueError(
            f"Unexpected {source_column} values {sorted(observed)}; "
            f"expected 1-based folds 1-20 or 0-based folds 0-19."
        )
    return result


def protocol_splits(df):
    """Return the four protocol partitions after fold normalization."""
    normalized = add_protocol_fold(df)
    return {
        "base_train": normalized[normalized["protocol_fold"].isin(BASE_TRAIN_FOLDS)].reset_index(drop=True),
        "qmodel": normalized[normalized["protocol_fold"].isin(QMODEL_FOLDS)].reset_index(drop=True),
        "validation": normalized[normalized["protocol_fold"] == VALIDATION_FOLD].reset_index(drop=True),
        "test": normalized[normalized["protocol_fold"] == TEST_FOLD].reset_index(drop=True),
    }


def select_threshold_for_sensitivity(probabilities, labels, thresholds, target=SENSITIVITY_TARGET):
    """Select the most specific threshold whose sensitivity reaches target."""
    best = None
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        positives = labels == 1
        negatives = labels == 0
        tp = int((predictions[positives] == 1).sum())
        fn = int((predictions[positives] == 0).sum())
        tn = int((predictions[negatives] == 0).sum())
        fp = int((predictions[negatives] == 1).sum())
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        candidate = (specificity, threshold, sensitivity)
        if sensitivity >= target and (best is None or candidate > best):
            best = candidate
    if best is None:
        raise ValueError(f"No threshold reaches sensitivity >= {target:.2f}.")
    return float(best[1]), float(best[2])