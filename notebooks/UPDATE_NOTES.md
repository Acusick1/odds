# ML Strategy Notebook - Update Notes

## Feature Extractor Refactor (2025-11-07)

The notebook uses `FeatureEngineering` which is now **deprecated but still functional**. The new recommended approach uses pluggable feature extractors via dependency injection.

### What Changed

- Feature extraction logic moved to `analytics.feature_extraction` module
- New `TabularFeatureExtractor` class replaces `FeatureEngineering`
- `XGBoostStrategy` now accepts feature extractors via dependency injection
- Full backward compatibility maintained (old code still works with deprecation warnings)

### Quick Migration Guide

**Old pattern (still works):**
```python
from analytics.ml_strategy_example import FeatureEngineering

features = FeatureEngineering.extract_features(event, odds, outcome="Lakers")
```

**New pattern (recommended):**
```python
from analytics.feature_extraction import TabularFeatureExtractor

extractor = TabularFeatureExtractor()
features = extractor.extract_features(event, odds, outcome="Lakers")
```

**Strategy initialization:**
```python
# Old (still works)
strategy = XGBoostStrategy()

# New (with custom extractor)
from analytics.feature_extraction import TabularFeatureExtractor
extractor = TabularFeatureExtractor(
    sharp_bookmakers=["pinnacle", "circa"],
    retail_bookmakers=["fanduel", "draftkings"]
)
strategy = XGBoostStrategy(feature_extractor=extractor)
```

### Why This Change?

1. **Extensibility**: Easy to support different model types (LSTM, Transformers) with different feature needs
2. **Testability**: Feature extraction can be tested independently
3. **Flexibility**: Customize bookmaker lists, add custom features
4. **Clean architecture**: Decouples feature engineering from strategy logic

### For More Information

See `notebooks/FEATURE_EXTRACTOR_GUIDE.md` for comprehensive documentation on:
- Using the new feature extractor API
- Implementing custom extractors
- Future LSTM/Transformer support
- Best practices and testing

### Migration Timeline

- **Current notebook**: Works as-is with deprecation warnings
- **Recommended action**: Update to new pattern when convenient
- **Breaking change**: None - full backward compatibility maintained
