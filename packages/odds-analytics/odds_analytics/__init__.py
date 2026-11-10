"""
Analytics and ML functionality for betting odds pipeline.

Provides candidate scoring and replay, feature extraction, model training and backfill.

Heavy dependencies (mlflow, scikit-learn, etc.) are optional —
install with ``pip install odds-analytics[training]`` for full functionality.
Modules that require training extras raise ImportError at import time if missing.
"""
