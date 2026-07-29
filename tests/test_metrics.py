import numpy as np

from ccvr.metrics import monotone_score, rank_metrics, soft_logic


def test_rank_metrics_and_constraint_violation() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6])
    labels = np.asarray([1, 0, 1, 0], dtype=np.int8)
    result = rank_metrics(scores, labels, np.asarray([10, 11, 12, 13]), {11})
    assert result["map"] == (1.0 + 2.0 / 3.0) / 2.0
    assert result["constraint_violation_at_10"] == 0.5
    assert result["near_miss_at_10"] == 0.25


def test_soft_logic_is_commutative_and_monotone() -> None:
    values = np.asarray([[0.2, 0.7], [0.8, 0.4]])
    swapped = values[::-1]
    for operator in ("AND", "OR"):
        first = soft_logic(values, operator, tau=0.1)
        second = soft_logic(swapped, operator, tau=0.1)
        assert np.allclose(first, second)
        increased = values.copy()
        increased[0] += 0.1
        assert np.all(
            soft_logic(increased, operator, tau=0.1) >= first - 1e-12
        )


def test_monotone_score_respects_positive_and_negated_satisfaction() -> None:
    base = np.asarray([0.1, 0.1])
    literals = np.asarray([[0.2, 0.8], [0.4, 0.4]])
    scores = monotone_score(base, literals, "AND", base_weight=0.5, tau=0.1)
    assert scores[1] > scores[0]
