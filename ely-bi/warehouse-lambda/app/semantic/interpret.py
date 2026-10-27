"""Read a business question against one catalog card.

The reading names the period, the measures, the people, and any cutoffs.
It does not know column codes. Those stay on the catalog objects it returns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import TypeVar

from app.semantic.catalog import GroupBy, Metric, PhraseBinding, Population, RoleField, TableCard
from app.semantic.format import contains_phrase, normalize

_TOP = re.compile(r"\btop\s+(\d+)\b", re.I)
_SHARE = re.compile(r"\bshare\b", re.I)
_OFFICE_TOTAL = re.compile(r"\boffice total\b", re.I)
_TOTAL = re.compile(r"\btotal\b", re.I)
_AVERAGE = re.compile(r"\b(average|avg|mean)\b", re.I)
_WHO = re.compile(r"\b(who are|who is|list)\b", re.I)
_EXTREME = re.compile(r"\b(highest|most|lowest|least)\b", re.I)
_COMPARISON = re.compile(
    r"\b(above|over|greater than|more than|below|under|less than|at least|at most)\s+"
    r"(-?\d+(?:\.\d+)?)\s*(%?)",
    re.I,
)
_EQUALITY = re.compile(r"\bof\s+(-?\d+(?:\.\d+)?)\s*(%?)", re.I)
_Value = TypeVar("_Value")

_OPS = {
    "above": ">",
    "over": ">",
    "greater than": ">",
    "more than": ">",
    "below": "<",
    "under": "<",
    "less than": "<",
    "at least": ">=",
    "at most": "<=",
}


@dataclass(frozen=True)
class Comparison:
    """One cutoff already scaled to the measure's stored units."""

    metric: Metric
    op: str
    bound: Decimal
    label: str


@dataclass(frozen=True)
class Reading:
    """What the question is asking for, in catalog terms."""

    grain: str
    metrics: tuple[Metric, ...]
    comparisons: tuple[Comparison, ...]
    population: Population | None
    person: str | None
    named_people: tuple[str, ...]
    person_is_office: bool
    designation: str | None
    group: GroupBy | None
    group_values: tuple[str, ...]
    roles: tuple[RoleField, ...]
    extreme: str
    top_n: int | None
    share: bool
    average: bool
    office_total: bool
    who: bool
    channel: tuple[str, str] | None
    show_companions: bool

    def kind(self) -> str:
        """The query shape, or an empty string when the catalog cannot answer it."""
        if self.roles and (self.who or self.person or self.person_is_office):
            return "hierarchy"
        if self.share and self.top_n and self.metrics:
            return "share"
        if self.office_total and self.metrics:
            return "office_total"
        if self.comparisons and self.metrics:
            return "threshold"
        if self.average and self.metrics:
            return "average"
        if self.extreme and self.metrics:
            return "extreme"
        if self.top_n and self.metrics:
            return "top"
        if self.designation and self.metrics:
            return "rank"
        if self.person and self.metrics and not self.person_is_office:
            return "person"
        if len(self.named_people) > 1 and self.metrics:
            return "people"
        if self.who and (self.designation or self.population):
            return "list"
        return ""


def interpret(
    question: str,
    card: TableCard,
    *,
    people: tuple[str, ...],
    designations: tuple[str, ...],
    group_values: dict[str, tuple[str, ...]],
    office: str,
    hierarchy: TableCard | None,
    phrases: tuple[PhraseBinding, ...] = (),
) -> Reading:
    """Build a reading from the question and the values that exist in the figures."""
    text = normalize(question)
    spans = _metric_spans(text, card.metrics)
    metrics = tuple(dict.fromkeys(metric for _, _, metric in spans))
    person, named_people, person_is_office = _person(text, people, office)
    group, chosen_groups = _group(card, text, group_values)
    # "Total paid NOP" is the office total. "Office total" already says so.
    # A named person keeps the person reading even if the question says total.
    office_total = bool(_OFFICE_TOTAL.search(text)) or (
        bool(_TOTAL.search(text)) and bool(metrics) and person is None and not person_is_office
    )
    return Reading(
        grain=_grain(card, text),
        metrics=metrics,
        comparisons=_comparisons(text, spans),
        population=_population(card, text, phrases),
        person=person,
        named_people=named_people,
        person_is_office=person_is_office,
        designation=_designation(text, designations),
        group=group,
        group_values=chosen_groups,
        roles=_roles(hierarchy, text),
        extreme=_extreme(text),
        top_n=_top_n(text),
        share=bool(_SHARE.search(text)),
        average=bool(_AVERAGE.search(text)),
        office_total=office_total,
        who=bool(_WHO.search(text)),
        channel=_channel(card, text),
        show_companions=contains_phrase(text, "count") or contains_phrase(text, "behind"),
    )


def _person(
    text: str,
    people: tuple[str, ...],
    office: str,
) -> tuple[str | None, tuple[str, ...], bool]:
    matches = _longest_contained(text, people)
    matches.sort(key=lambda value: text.find(normalize(value)))
    if len(matches) != 1:
        return None, tuple(matches), False
    name = matches[0]
    is_office = bool(office) and normalize(name) == normalize(office)
    return name, (), is_office


def _designation(text: str, designations: tuple[str, ...]) -> str | None:
    best = ""
    best_len = 0
    for designation in designations:
        token = normalize(designation)
        if not token:
            continue
        plural = token if token.endswith("s") else f"{token}s"
        if contains_phrase(text, token) or contains_phrase(text, plural):
            if len(token) > best_len:
                best = designation
                best_len = len(token)
    return best or None


def _group(
    card: TableCard,
    text: str,
    group_values: dict[str, tuple[str, ...]],
) -> tuple[GroupBy | None, tuple[str, ...]]:
    best: GroupBy | None = None
    chosen: list[str] = []
    for group in card.groups:
        values = group_values.get(group.lookup, ())
        hits = [
            value
            for _, value in sorted(
                (text.find(normalize(value)), value)
                for value in values
                if value and contains_phrase(text, normalize(value))
            )
        ]
        mentioned = any(contains_phrase(text, normalize(phrase)) for phrase in group.phrases)
        if not hits and not mentioned:
            continue
        if best is None or len(hits) > len(chosen):
            best = group
            chosen = hits
    return best, tuple(chosen)


def _roles(card: TableCard | None, text: str) -> tuple[RoleField, ...]:
    if card is None:
        return ()
    found = [
        role
        for role in card.roles
        if any(contains_phrase(text, normalize(phrase)) for phrase in role.phrases)
    ]
    return tuple(found)


def _longest_contained(text: str, values: tuple[str, ...]) -> list[str]:
    hits = [value for value in values if value and contains_phrase(text, normalize(value))]
    hits.sort(key=lambda value: len(normalize(value)), reverse=True)
    kept: list[str] = []
    for value in hits:
        token = normalize(value)
        if any(token != normalize(other) and token in normalize(other) for other in kept):
            continue
        kept.append(value)
    return kept


def _metric_spans(text: str, metrics: tuple[Metric, ...]) -> list[tuple[int, int, Metric]]:
    phrases: list[tuple[str, Metric]] = []
    for metric in metrics:
        for phrase in metric.phrases:
            phrases.append((normalize(phrase), metric))
    phrases.sort(key=lambda item: len(item[0]), reverse=True)
    occupied = [False] * (len(text) + 1)
    found: list[tuple[int, int, Metric]] = []
    for phrase, metric in phrases:
        start = 0
        while phrase:
            index = text.find(phrase, start)
            if index < 0:
                break
            end = index + len(phrase)
            if not any(occupied[index:end]) and _bounded(text, index, end):
                for cursor in range(index, end):
                    occupied[cursor] = True
                found.append((index, end, metric))
            start = index + 1
    found.sort(key=lambda item: item[0])
    return found


def _bounded(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start else " "
    after = text[end] if end < len(text) else " "
    return not before.isalnum() and not after.isalnum()


_MAX_COMPARISON_GAP = 80


def _comparisons(text: str, spans: list[tuple[int, int, Metric]]) -> tuple[Comparison, ...]:
    found: list[tuple[int, Comparison]] = []
    for match in _COMPARISON.finditer(text):
        metric = _nearest_metric(spans, match.start())
        if metric is None:
            continue
        number = Decimal(match.group(2))
        found.append(
            (
                match.start(),
                Comparison(
                    metric=metric,
                    op=_OPS[match.group(1).lower()],
                    bound=_scale_bound(metric, number),
                    label=" ".join(match.group(0).split()),
                ),
            )
        )
    for match in _EQUALITY.finditer(text):
        metric = _nearest_metric(spans, match.start())
        if metric is None:
            continue
        number = Decimal(match.group(1))
        suffix = "%" if match.group(2) or metric.scale in {"ratio", "percent"} else ""
        found.append(
            (
                match.start(),
                Comparison(
                    metric=metric,
                    op="=",
                    bound=_scale_bound(metric, number),
                    label=f"equal to {match.group(1)}{suffix}",
                ),
            )
        )
    found.sort(key=lambda item: item[0])
    unique: list[Comparison] = []
    seen: set[tuple[str, str, Decimal]] = set()
    for _, item in found:
        key = (item.metric.id, item.op, item.bound)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return tuple(unique)


def _nearest_metric(spans: list[tuple[int, int, Metric]], index: int) -> Metric | None:
    """Bind a cutoff to the measure beside it, before or after the number.

    A cutoff with no measure within a short window is left unbound so a
    distant number is not applied to the wrong figure.
    """
    chosen: Metric | None = None
    distance = _MAX_COMPARISON_GAP + 1
    for start, end, metric in spans:
        if start <= index <= end:
            return metric
        gap = min(abs(index - start), abs(index - end))
        if gap < distance:
            chosen = metric
            distance = gap
    if distance > _MAX_COMPARISON_GAP:
        return None
    return chosen


def _scale_bound(metric: Metric, number: Decimal) -> Decimal:
    """Convert a spoken percent into the units stored for that measure."""
    if metric.scale == "ratio" and abs(number) > 1:
        return number / Decimal("100")
    return number


def _population(
    card: TableCard,
    text: str,
    phrases: tuple[PhraseBinding, ...] = (),
) -> Population | None:
    options: list[tuple[tuple[str, ...], Population]] = []
    for population in card.populations:
        if population.id == "office_total":
            continue
        aliases = tuple(
            phrase
            for binding in phrases
            if binding.population == population.id
            for phrase in binding.phrases
        )
        options.append((population.phrases + aliases, population))
    return _last_value(text, options)


def _channel(card: TableCard, text: str) -> tuple[str, str] | None:
    options = [
        (phrases, (column, equals))
        for phrases, column, equals in card.channels
    ]
    return _last_value(text, options)


def _grain(card: TableCard, text: str) -> str:
    chosen = _last_value(text, list(card.grains))
    return chosen or card.default_grain or "MTD"


def _last_value(
    text: str,
    options: list[tuple[tuple[str, ...], _Value]],
) -> _Value | None:
    """The value whose phrase occurs latest. A longer phrase wins the same spot.

    A follow-up can change the period or the group. The latest phrase in the
    conversation is the one the person means now.
    """
    best = None
    best_at = -1
    best_len = -1
    for phrases, value in options:
        for phrase in phrases:
            token = normalize(phrase)
            if not token:
                continue
            start = 0
            while True:
                index = text.find(token, start)
                if index < 0:
                    break
                end = index + len(token)
                if _bounded(text, index, end) and (
                    index > best_at or (index == best_at and len(token) > best_len)
                ):
                    best = value
                    best_at = index
                    best_len = len(token)
                start = index + 1
    return best


def _extreme(text: str) -> str:
    match = _EXTREME.search(text)
    if not match:
        return ""
    word = match.group(1).lower()
    if word in {"highest", "most"}:
        return "highest"
    return "lowest"


def _top_n(text: str) -> int | None:
    match = _TOP.search(text)
    if not match:
        return None
    return max(1, min(int(match.group(1)), 25))
