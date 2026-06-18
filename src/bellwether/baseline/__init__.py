"""Baseline manager: learns and stores per-(agent, task_class, fingerprint) baselines.

Implements design doc 04's composable representation: a sliding window of robust per-feature
estimators plus categorical frequency tables, conditioned by context, keyed by baseline lineage
(design doc 03). Cold-start/warmup is handled at scoring time via ``min_samples``.
"""

from bellwether.baseline.manager import Baseline, BaselineManager

__all__ = ["Baseline", "BaselineManager"]
