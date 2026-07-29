from ccvr.logic import (
    evaluate_expression,
    operator_aware_near_misses,
    parse_expression,
)


def test_parse_and_evaluate_signed_boolean_expressions() -> None:
    atomic = {
        ("+", "red"): {1, 2},
        ("-", "red"): {3, 4},
        ("+", "indoor"): {1, 3},
        ("-", "indoor"): {2, 4},
    }
    conjunction = parse_expression("AND +red -indoor")
    disjunction = parse_expression("OR +red +indoor")
    assert evaluate_expression(conjunction, atomic) == {2}
    assert evaluate_expression(disjunction, atomic) == {1, 2, 3}
    assert conjunction.contains_negation


def test_operator_aware_near_misses_are_logically_valid() -> None:
    atomic = {
        ("+", "red"): {1, 2},
        ("+", "indoor"): {1, 3},
    }
    conjunction = parse_expression("AND +red +indoor")
    positives = evaluate_expression(conjunction, atomic)
    near, definition = operator_aware_near_misses(
        conjunction, positives, {1, 2, 3, 4}, atomic
    )
    assert positives == {1}
    assert near == {2, 3}
    assert definition == "satisfies_some_but_not_all_literals"

    disjunction = parse_expression("OR +red +indoor")
    positives = evaluate_expression(disjunction, atomic)
    near, definition = operator_aware_near_misses(
        disjunction, positives, {1, 2, 3, 4}, atomic
    )
    assert positives == {1, 2, 3}
    assert near == {4}
    assert definition == "base_relevant_and_satisfies_no_disjunct"


def test_boolean_operators_are_commutative() -> None:
    atomic = {
        ("+", "a"): {1, 2},
        ("-", "b"): {2, 3},
    }
    for operator in ("AND", "OR"):
        left = evaluate_expression(parse_expression(f"{operator} +a -b"), atomic)
        right = evaluate_expression(parse_expression(f"{operator} -b +a"), atomic)
        assert left == right

