"""Feature validation utilities."""

import numpy as np

from analytics.core import FeatureSet, SequentialFeatureSet, TabularFeatureSet


def validate_no_inf(features: FeatureSet) -> tuple[bool, list[str]]:
    """Check for infinite values in features.

    Args:
        features: Feature set to validate

    Returns:
        Tuple of (is_valid, list of problematic feature names)
    """
    problems = []

    if isinstance(features, TabularFeatureSet):
        for name, value in features.features.items():
            if np.isinf(value):
                problems.append(f"{name}=inf")

    elif isinstance(features, SequentialFeatureSet):
        if np.any(np.isinf(features.sequences)):
            problems.append("sequential features contain inf")

    return len(problems) == 0, problems


def validate_no_nan(features: FeatureSet) -> tuple[bool, list[str]]:
    """Check for NaN values in features.

    Args:
        features: Feature set to validate

    Returns:
        Tuple of (is_valid, list of problematic feature names)
    """
    problems = []

    if isinstance(features, TabularFeatureSet):
        for name, value in features.features.items():
            if np.isnan(value):
                problems.append(f"{name}=nan")

    elif isinstance(features, SequentialFeatureSet):
        if np.any(np.isnan(features.sequences)):
            problems.append("sequential features contain nan")

    return len(problems) == 0, problems


def validate_feature_range(
    features: TabularFeatureSet,
    expected_ranges: dict[str, tuple[float, float]],
) -> tuple[bool, list[str]]:
    """Validate that features fall within expected ranges.

    Args:
        features: Tabular feature set
        expected_ranges: Dict mapping feature name to (min, max) tuple

    Returns:
        Tuple of (is_valid, list of out-of-range features)
    """
    problems = []

    for name, (min_val, max_val) in expected_ranges.items():
        if name not in features.features:
            continue

        value = features.features[name]
        if value < min_val or value > max_val:
            problems.append(f"{name}={value} not in [{min_val}, {max_val}]")

    return len(problems) == 0, problems


def validate_all(features: FeatureSet) -> tuple[bool, list[str]]:
    """Run all standard validations.

    Args:
        features: Feature set to validate

    Returns:
        Tuple of (is_valid, list of all problems)
    """
    all_problems = []

    # Check for inf
    valid, problems = validate_no_inf(features)
    if not valid:
        all_problems.extend(problems)

    # Check for nan
    valid, problems = validate_no_nan(features)
    if not valid:
        all_problems.extend(problems)

    return len(all_problems) == 0, all_problems
