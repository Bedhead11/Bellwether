"""Tests for the skill tier: focused signature scoring + improvement on a weak fault."""

from bellwether.baseline import BaselineManager
from bellwether.detect import DriftScorer, ScoringConfig
from bellwether.detect.engine import calibrate_threshold
from bellwether.detect.report import SubScore
from bellwether.features import FeatureFamily
from bellwether.fixtures import FaultSpec, FixtureAgent, faulted_run
from bellwether.improve import DriftSignature, SkillLibrary

F = FeatureFamily


def _sub(
    feature: str, context: str, score: float, direction: float, magnitude: float | None = None
) -> SubScore:
    mag = abs(direction) if magnitude is None else magnitude
    return SubScore(feature, context, F.ECONOMIC, score, 1 - score, direction, magnitude=mag)


# --- focused scoring ---------------------------------------------------------------------


def test_focused_score_uses_only_matched_features() -> None:
    sig = DriftSignature("cost", (("step_input_tokens", "llm", 1), ("step_cost", "llm", 1)))
    subs = [
        _sub("step_input_tokens", "llm", 0.99, +12.0),  # matches, strong anomaly up
        _sub("step_cost", "llm", 0.97, +9.0),  # matches, right direction
        _sub("step_latency", "llm", 0.99, +12.0),  # not in signature -> ignored
    ]
    score = sig.focused_score(subs)
    assert score > 0.9  # sharp: a large matched magnitude squashes near 1


def test_direction_filter_excludes_wrong_direction() -> None:
    sig = DriftSignature("cost_up", (("step_input_tokens", "llm", 1),))
    # tokens went DOWN but signature expects UP -> no match -> zero focused score
    subs = [_sub("step_input_tokens", "llm", 0.99, -12.0)]
    assert sig.focused_score(subs) == 0.0


def test_focused_score_monotone_in_magnitude() -> None:
    sig = DriftSignature("cost", (("step_cost", "llm", 1),))
    weak = sig.focused_score([_sub("step_cost", "llm", 0.9, +3.0)])
    strong = sig.focused_score([_sub("step_cost", "llm", 0.99, +20.0)])
    assert strong > weak  # bigger anomaly -> sharper focused score (no flooring collision)


def test_library_serialization_roundtrip() -> None:
    lib = SkillLibrary()
    lib.add(DriftSignature("a", (("f", "llm", 1),), description="d"))
    lib.add(DriftSignature("b", (("g", "*", -1),)))
    restored = SkillLibrary.from_dict(lib.to_dict())
    assert restored.names == ["a", "b"]
    assert restored.signatures[0].description == "d"
    assert restored.signatures[1].feature_directions == (("g", "*", -1),)


# --- integration: a signature improves a weak fault without raising FP -------------------


def _train(agent: FixtureAgent, n: int = 120):  # type: ignore[no-untyped-def]
    mgr = BaselineManager()
    for s in range(n):
        mgr.learn(agent.clean_run(seed=s))
    return mgr.baseline_for(agent.clean_run(seed=0))


def _cost_recall_and_fp(scorer: DriftScorer, agent: FixtureAgent, baseline):  # type: ignore[no-untyped-def]
    cal = [agent.clean_run(seed=s) for s in range(120, 270)]
    thr = calibrate_threshold(scorer, baseline, cal, target_fp_rate=0.02)
    timely = 0
    for seed in range(300, 320):
        run = faulted_run(agent, seed, FaultSpec("cost_blowup", 0.7, onset_step=1))
        rep = scorer.evaluate(run, baseline, thr)
        if rep.alert.triggered and rep.alert.step_index is not None:
            vf = run.injected_fault.visible_failure_step  # type: ignore[union-attr]
            if rep.alert.step_index <= vf:
                timely += 1
    held_out = [agent.clean_run(seed=s) for s in range(400, 440)]
    fp = sum(scorer.evaluate(r, baseline, thr).alert.triggered for r in held_out)
    return timely / 20, fp / len(held_out)


def test_cost_signature_improves_timely_recall_without_raising_fp() -> None:
    agent = FixtureAgent()
    baseline = _train(agent)
    cfg = ScoringConfig(min_samples=30)

    generic = DriftScorer(cfg)
    gen_recall, gen_fp = _cost_recall_and_fp(generic, agent, baseline)

    lib = SkillLibrary(
        [
            DriftSignature(
                "cost_blowup",
                (("step_input_tokens", "llm", 1), ("step_cost", "llm", 1)),
                description="prompt/cost growth on LLM steps",
            )
        ]
    )
    skilled = DriftScorer(cfg, library=lib)
    skill_recall, skill_fp = _cost_recall_and_fp(skilled, agent, baseline)

    assert skill_recall >= gen_recall  # the signature helps (or at least never hurts)
    assert skill_recall > gen_recall or gen_recall == 1.0  # expect a strict gain on this weak fault
    assert skill_fp <= 0.10  # false positives stay controlled


def test_signature_attribution_names_the_signature() -> None:
    agent = FixtureAgent()
    baseline = _train(agent)
    lib = SkillLibrary(
        [DriftSignature("cost_blowup", (("step_input_tokens", "llm", 1), ("step_cost", "llm", 1)))]
    )
    scorer = DriftScorer(ScoringConfig(min_samples=30), library=lib)
    cal = [agent.clean_run(seed=s) for s in range(120, 270)]
    thr = calibrate_threshold(scorer, baseline, cal, target_fp_rate=0.02)

    fired = False
    for seed in range(300, 320):
        run = faulted_run(agent, seed, FaultSpec("cost_blowup", 0.9, onset_step=1))
        rep = scorer.evaluate(run, baseline, thr)
        if rep.alert.signature == "cost_blowup":
            fired = True
            break
    assert fired
