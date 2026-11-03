import numpy as np

from shelfspace.experiment import (
    minimum_detectable_effect,
    power,
    required_n_per_arm,
    simulate_experiment,
)


def test_aa_test_false_positive_rate_is_about_five_percent():
    hits = [simulate_experiment(0.0, seed=s)["significant"] for s in range(2000)]
    assert 0.035 < np.mean(hits) < 0.065


def test_analytic_power_matches_simulation():
    n, uplift = 20, 0.05
    sim = np.mean(
        [simulate_experiment(uplift, n_per_arm=n, seed=s)["significant"] for s in range(1500)]
    )
    assert abs(sim - power(uplift, n)) < 0.06


def test_confidence_interval_coverage():
    covered = []
    for s in range(1000):
        r = simulate_experiment(0.03, seed=s)
        covered.append(r["ci_low"] <= 0.03 <= r["ci_high"])
    assert 0.92 < np.mean(covered) < 0.98


def test_required_sample_size_is_consistent_with_power():
    n = required_n_per_arm(0.02, target_power=0.8)
    assert power(0.02, n) >= 0.8
    assert power(0.02, n - 1) < 0.8
    assert required_n_per_arm(0.01) > required_n_per_arm(0.05)


def test_minimum_detectable_effect_shrinks_with_more_stores():
    assert minimum_detectable_effect(100) < minimum_detectable_effect(10)
    assert abs(power(minimum_detectable_effect(30), 30) - 0.8) < 1e-6
