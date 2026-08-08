from __future__ import annotations

from collections import Counter
import math
import random
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


LOG2 = math.log(2.0)


def _normalize_source_value(value: Any) -> Tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if probability <= 0:
        return sorted_values[0]
    if probability >= 1:
        return sorted_values[-1]

    index = (len(sorted_values) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def mutual_information(
    source_samples: Sequence[Any],
    target_samples: Sequence[Any],
    *,
    smoothing_alpha: float = 0.0,
    estimator: str = "plugin",
) -> float:
    if len(source_samples) != len(target_samples):
        raise ValueError("source_samples and target_samples must be the same length")
    if not source_samples:
        raise ValueError("At least one sample is required")
    if smoothing_alpha < 0:
        raise ValueError("smoothing_alpha must be non-negative")
    if estimator not in {"plugin", "miller_madow"}:
        raise ValueError(f"Unsupported estimator: {estimator}")

    normalized_sources = [_normalize_source_value(value) for value in source_samples]
    source_counts = Counter(normalized_sources)
    target_counts = Counter(target_samples)
    joint_counts = Counter(zip(normalized_sources, target_samples))
    sample_count = len(normalized_sources)

    observed_sources = list(source_counts.keys())
    observed_targets = list(target_counts.keys())
    total_joint_cells = len(observed_sources) * len(observed_targets)
    smoothed_total = sample_count + smoothing_alpha * total_joint_cells

    mi_bits = 0.0
    for source_value in observed_sources:
        source_prob = (
            source_counts[source_value] + smoothing_alpha * len(observed_targets)
        ) / smoothed_total
        for target_value in observed_targets:
            joint_prob = (joint_counts[(source_value, target_value)] + smoothing_alpha) / smoothed_total
            if joint_prob <= 0:
                continue
            target_prob = (
                target_counts[target_value] + smoothing_alpha * len(observed_sources)
            ) / smoothed_total
            mi_bits += joint_prob * math.log2(joint_prob / (source_prob * target_prob))

    if estimator == "miller_madow":
        k_source = len(observed_sources)
        k_target = len(observed_targets)
        k_joint = len(joint_counts)
        correction_bits = ((k_source - 1) + (k_target - 1) - (k_joint - 1)) / (2.0 * sample_count * LOG2)
        mi_bits += correction_bits

    return max(mi_bits, 0.0)


def estimate_mi_from_records(
    records: Sequence[Dict[str, Any]],
    *,
    source_keys: Sequence[str],
    target_key: str,
    smoothing_alpha: float = 0.0,
    estimator: str = "plugin",
) -> float:
    source_samples = [tuple(record[key] for key in source_keys) for record in records]
    target_samples = [record[target_key] for record in records]
    return mutual_information(
        source_samples,
        target_samples,
        smoothing_alpha=smoothing_alpha,
        estimator=estimator,
    )


def smoothing_sensitivity(
    records: Sequence[Dict[str, Any]],
    *,
    source_keys: Sequence[str],
    target_key: str,
    alphas: Sequence[float],
    estimator: str = "plugin",
) -> List[Dict[str, float]]:
    results = []
    for alpha in alphas:
        results.append(
            {
                "alpha": alpha,
                "mi_bits": estimate_mi_from_records(
                    records,
                    source_keys=source_keys,
                    target_key=target_key,
                    smoothing_alpha=alpha,
                    estimator=estimator,
                ),
            }
        )
    return results


def bootstrap_confidence_interval(
    records: Sequence[Dict[str, Any]],
    *,
    source_keys: Sequence[str],
    target_key: str,
    estimator: str = "plugin",
    smoothing_alpha: float = 0.0,
    iterations: int = 500,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    if not records:
        raise ValueError("At least one record is required")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    rng = random.Random(seed)
    estimates: List[float] = []
    sample_size = len(records)
    for _ in range(iterations):
        sampled_records = [records[rng.randrange(sample_size)] for _ in range(sample_size)]
        estimates.append(
            estimate_mi_from_records(
                sampled_records,
                source_keys=source_keys,
                target_key=target_key,
                estimator=estimator,
                smoothing_alpha=smoothing_alpha,
            )
        )

    estimates.sort()
    alpha = 1.0 - confidence_level
    return {
        "confidence_level": confidence_level,
        "iterations": iterations,
        "point_estimate": estimate_mi_from_records(
            records,
            source_keys=source_keys,
            target_key=target_key,
            estimator=estimator,
            smoothing_alpha=smoothing_alpha,
        ),
        "lower_bound": _percentile(estimates, alpha / 2.0),
        "upper_bound": _percentile(estimates, 1.0 - (alpha / 2.0)),
        "bootstrap_mean": mean(estimates),
        "bootstrap_estimates": estimates,
    }


def permutation_null_distribution(
    records: Sequence[Dict[str, Any]],
    *,
    source_keys: Sequence[str],
    target_key: str,
    estimator: str = "plugin",
    smoothing_alpha: float = 0.0,
    iterations: int = 250,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    if not records:
        raise ValueError("At least one record is required")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    rng = random.Random(seed)
    targets = [record[target_key] for record in records]
    shuffled_estimates: List[float] = []
    point_estimate = estimate_mi_from_records(
        records,
        source_keys=source_keys,
        target_key=target_key,
        estimator=estimator,
        smoothing_alpha=smoothing_alpha,
    )
    for _ in range(iterations):
        shuffled_targets = list(targets)
        rng.shuffle(shuffled_targets)
        shuffled_records = []
        for index, record in enumerate(records):
            shuffled_record = dict(record)
            shuffled_record[target_key] = shuffled_targets[index]
            shuffled_records.append(shuffled_record)
        shuffled_estimates.append(
            estimate_mi_from_records(
                shuffled_records,
                source_keys=source_keys,
                target_key=target_key,
                estimator=estimator,
                smoothing_alpha=smoothing_alpha,
            )
        )

    exceedances = sum(1 for value in shuffled_estimates if value >= point_estimate)
    p_value = (exceedances + 1) / (iterations + 1)
    return {
        "iterations": iterations,
        "point_estimate": point_estimate,
        "null_estimates": shuffled_estimates,
        "p_value": p_value,
    }