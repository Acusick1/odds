# Feature Extractor Architecture Guide

## Overview

As of the feature extraction refactor, the ML strategy system now uses **pluggable feature extractors** via dependency injection. This enables clean support for different model architectures (XGBoost, LSTM, Transformers) that require fundamentally different feature extraction approaches.

## Architecture

### Core Components

1. **FeatureExtractor** (abstract base class): Defines the interface all extractors must implement
2. **TabularFeatureExtractor**: Snapshot-based features for tabular models (XGBoost, Random Forest)
3. **SequenceFeatureExtractor**: Time-series features for sequence models (LSTM, Transformers) - *stub for future implementation*

### Why This Design?

- **Decoupling**: Feature engineering is separated from betting strategy logic
- **Extensibility**: Easy to add new feature extractors for different model types
- **Testability**: Feature extraction can be tested independently
- **Flexibility**: Mix and match extractors with strategies

## Using TabularFeatureExtractor

### Default Usage (Recommended)

```python
from analytics.feature_extraction import TabularFeatureExtractor
from analytics.ml_strategy_example import XGBoostStrategy

# Create extractor with default settings
extractor = TabularFeatureExtractor()

# Inject into strategy (or use default)
strategy = XGBoostStrategy(feature_extractor=extractor)
```

### Custom Bookmaker Configuration

```python
# Custom sharp bookmakers (e.g., Pinnacle + Circa)
extractor = TabularFeatureExtractor(
    sharp_bookmakers=["pinnacle", "circa"],
    retail_bookmakers=["fanduel", "draftkings"]
)

strategy = XGBoostStrategy(feature_extractor=extractor)
```

### Direct Feature Extraction

```python
# Extract features manually for analysis
features = extractor.extract_features(
    event=backtest_event,
    odds_data=odds_snapshot,
    outcome="Los Angeles Lakers",
    market="h2h"
)

# Convert to numpy array
feature_names = extractor.get_feature_names()
vector = extractor.create_feature_vector(features, feature_names)
```

## Migrating from FeatureEngineering (Deprecated)

### Old Pattern

```python
# Deprecated (still works but shows warning)
from analytics.ml_strategy_example import FeatureEngineering

features = FeatureEngineering.extract_features(event, odds, outcome="Lakers")
vector = FeatureEngineering.create_feature_vector(features, feature_names)
```

### New Pattern

```python
# Recommended approach
from analytics.feature_extraction import TabularFeatureExtractor

extractor = TabularFeatureExtractor()
features = extractor.extract_features(event, odds, outcome="Lakers")
vector = extractor.create_feature_vector(features)
```

## Implementing Custom Feature Extractors

### Example: Custom Tabular Extractor

```python
from analytics.feature_extraction import FeatureExtractor
import numpy as np

class MyCustomExtractor(FeatureExtractor):
    """Custom feature extractor with additional features."""

    def __init__(self):
        self._feature_names = ["custom_feature_1", "custom_feature_2"]

    def extract_features(self, event, odds_data, outcome=None, **kwargs):
        """Extract custom features."""
        features = {}

        # Your feature engineering logic here
        features["custom_feature_1"] = 1.0
        features["custom_feature_2"] = 2.0

        return features

    def get_feature_names(self):
        """Return ordered list of feature names."""
        return self._feature_names

# Use with XGBoostStrategy
strategy = XGBoostStrategy(feature_extractor=MyCustomExtractor())
```

## Future: Sequence Features for LSTMs

The `SequenceFeatureExtractor` is a stub for future LSTM/Transformer implementations:

```python
from analytics.feature_extraction import SequenceFeatureExtractor

# Future usage (not yet implemented)
extractor = SequenceFeatureExtractor(
    lookback_hours=72,  # 72 hours of historical data
    timesteps=24        # Sample every 3 hours
)

# Will extract time-series sequences instead of single snapshots
sequence_features = extractor.extract_features(
    event=event,
    odds_data=odds_history,  # List of snapshots over time
    outcome="Lakers"
)
# Returns: {"sequence": np.ndarray(24, features), "mask": np.ndarray(24,)}
```

### Planned LSTM Features

- Line movement patterns over 72 hours
- Time-to-game encoding
- Sharp money indicators (reverse line movement)
- Steam moves (rapid line changes)
- Betting percentage vs line movement divergence

## Testing Feature Extractors

```python
# Test extraction produces valid features
extractor = TabularFeatureExtractor()
features = extractor.extract_features(event, odds, outcome="Lakers")

assert isinstance(features, dict)
assert len(features) > 0
assert all(isinstance(v, float) for v in features.values())

# Test vector conversion
feature_names = extractor.get_feature_names()
vector = extractor.create_feature_vector(features, feature_names)

assert isinstance(vector, np.ndarray)
assert len(vector) == len(feature_names)
assert np.all(np.isfinite(vector))  # No NaN or Inf
```

## Best Practices

1. **Use dependency injection**: Pass extractors to strategies rather than hard-coding
2. **Test independently**: Write unit tests for extractors separate from strategies
3. **Document features**: Maintain clear documentation of what each feature represents
4. **Feature consistency**: Ensure the same extractor is used for training and backtesting
5. **Save with models**: Consider serializing extractor config with trained models

## Integration with Training Workflow

### Recommended Pattern

```python
# 1. Create and configure extractor
extractor = TabularFeatureExtractor()

# 2. Extract training data
features_list = []
labels = []

for event, odds in zip(events, odds_snapshots):
    for outcome in [event.home_team, event.away_team]:
        features = extractor.extract_features(event, odds, outcome=outcome)
        if features:
            features_list.append(features)
            labels.append(1 if outcome_won else 0)

# 3. Convert to training arrays
features_df = pd.DataFrame(features_list)
feature_names = list(features_df.columns)
X_train = features_df.values
y_train = np.array(labels)

# 4. Train strategy with same extractor
strategy = XGBoostStrategy(feature_extractor=extractor)
strategy.train(X_train, y_train, feature_names)

# 5. Backtest with same extractor (automatically used)
result = await backtest_engine.run()
```

## See Also

- `analytics/feature_extraction.py` - Full implementation
- `tests/unit/test_feature_extraction.py` - Comprehensive tests
- `notebooks/ml_strategy_training.ipynb` - Training workflow example
- `CLAUDE.md` - System architecture documentation
