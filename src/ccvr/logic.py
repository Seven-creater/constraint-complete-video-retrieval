from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping


_LITERAL_PATTERN = re.compile(r"(?:^|\s)([+-])(.+?)(?=(?:\s+[+-])|$)")


@dataclass(frozen=True)
class Literal:
    tag: str
    polarity: str

    def __post_init__(self) -> None:
        if self.polarity not in {"+", "-"}:
            raise ValueError(f"unsupported polarity: {self.polarity}")
        if not self.tag.strip():
            raise ValueError("literal tag must not be empty")

    @property
    def key(self) -> tuple[str, str]:
        return self.polarity, self.tag

    def as_dict(self) -> dict[str, str]:
        return {"tag": self.tag, "polarity": self.polarity}


@dataclass(frozen=True)
class Expression:
    operator: str
    literals: tuple[Literal, ...]
    source: str

    def __post_init__(self) -> None:
        if self.operator not in {"ATOM", "AND", "OR"}:
            raise ValueError(f"unsupported operator: {self.operator}")
        expected = 1 if self.operator == "ATOM" else 2
        if len(self.literals) != expected:
            raise ValueError(
                f"{self.operator} expression requires {expected} literal(s), "
                f"got {len(self.literals)}: {self.source}"
            )

    @property
    def contains_negation(self) -> bool:
        return any(literal.polarity == "-" for literal in self.literals)


def parse_expression(value: str) -> Expression:
    source = str(value).strip()
    operator = "ATOM"
    body = source
    for candidate in ("AND", "OR"):
        prefix = candidate + " "
        if source.startswith(prefix):
            operator = candidate
            body = source[len(prefix) :]
            break
    literals = tuple(
        Literal(tag=match.group(2).strip(), polarity=match.group(1))
        for match in _LITERAL_PATTERN.finditer(body)
    )
    if not literals or "".join(
        f"{literal.polarity}{literal.tag}" for literal in literals
    ).replace(" ", "") != body.replace(" ", ""):
        raise ValueError(f"cannot parse expression: {source}")
    return Expression(operator=operator, literals=literals, source=source)


def evaluate_expression(
    expression: Expression,
    atomic_sets: Mapping[tuple[str, str], set[int]],
) -> set[int]:
    try:
        operands = [set(atomic_sets[literal.key]) for literal in expression.literals]
    except KeyError as exc:
        raise KeyError(
            f"missing atomic literal {exc.args[0]} for {expression.source}"
        ) from exc
    if expression.operator == "ATOM":
        return operands[0]
    if expression.operator == "AND":
        return set.intersection(*operands)
    return set.union(*operands)


def operator_aware_near_misses(
    expression: Expression,
    positives: set[int],
    base_relevant: Iterable[int],
    atomic_sets: Mapping[tuple[str, str], set[int]],
) -> tuple[set[int], str]:
    base = set(base_relevant)
    negatives = base - positives
    if expression.operator == "AND":
        satisfied_any = set().union(
            *(atomic_sets[literal.key] for literal in expression.literals)
        )
        return negatives & satisfied_any, "satisfies_some_but_not_all_literals"
    if expression.operator == "OR":
        return negatives, "base_relevant_and_satisfies_no_disjunct"
    return negatives, "base_relevant_and_violates_atomic_literal"

