"""Tests for topology-tier self-improvement (MAP-Elites over ensemble configs)."""

import random

from bellwether.detect import ScoringConfig
from bellwether.eval import BenchmarkConfig
from bellwether.fixtures import FixtureAgent
from bellwether.governance import AuditLog
from bellwether.improve import TopologySearch, descriptor, mutate_config


def test_mutate_keeps_config_valid() -> None:
    rng = random.Random(0)
    cfg = ScoringConfig()
    for _ in range(200):
        cfg = mutate_config(cfg, rng)
        assert 2 <= cfg.window_w <= 6
        assert 1 <= cfg.min_hits_k <= cfg.window_w
        assert 0.0 <= cfg.signature_budget_frac <= 0.6
        assert 15 <= cfg.min_samples <= 60
        fr = cfg.crit_budget_frac + cfg.step_budget_frac + cfg.run_budget_frac
        assert abs(fr - 1.0) < 1e-6
        assert min(cfg.crit_budget_frac, cfg.step_budget_frac, cfg.run_budget_frac) > 0


def test_descriptor_bins_by_window_and_dominant_track() -> None:
    cfg = ScoringConfig(window_w=4, crit_budget_frac=0.6, step_budget_frac=0.2, run_budget_frac=0.2)
    assert descriptor(cfg) == (4, "critical")


def test_topology_search_fills_archive_and_audits() -> None:
    agent = FixtureAgent()
    audit = AuditLog()
    cfg = BenchmarkConfig(
        n_train=50, n_cal=60, n_eval_benign=80, n_eval_per_fault=5, n_seeds=2, min_samples=20
    )
    search = TopologySearch(agent=agent, eval_config=cfg, audit=audit, seed=1)
    history = search.run(iterations=5)

    # The archive holds at least the seed niche; the best F1 curve never decreases (elitism).
    assert len(history.archive) >= 1
    curve = history.best_f1_curve()
    assert all(b >= a - 1e-9 for a, b in zip(curve, curve[1:], strict=False))
    # The best config is valid and the audit chain (if any acceptances) verifies.
    assert 2 <= history.best.config.window_w <= 6
    assert audit.verify()
