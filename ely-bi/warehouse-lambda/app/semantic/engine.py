"""Resolve a business question against the catalog, then read the figures.

A follow-up is returned only when the figures really contain more than one
match. A single office, "this month", and phrases such as "direct agency" are
bound here. Ratios, percents, and hyphenated periods are handled from the
catalog so a cutoff of 90% is not compared with a 0-1 ratio as if it were 90.
"""

from __future__ import annotations

import logging
import re
import textwrap
from dataclasses import replace

from app.semantic.catalog import (
    Catalog,
    PhraseBinding,
    TableCard,
    load_catalog,
    relevant_documents,
    relevant_tables,
)
from app.semantic.format import contains_phrase, name_list, normalize
from app.semantic.interpret import Reading, interpret
from app.semantic.queries import FigureResult, SqlRunner, execute

logger = logging.getLogger(__name__)

_GREETING = re.compile(r"(hi|hello|hey|thanks|thank you)[.! ]*", re.I)
_GRAIN_LABEL = {
    "MTD": "month to date",
    "QTD": "quarter to date",
    "YTD": "year to date",
    "FTD": "for the day",
}

__all__ = ["FigureResult", "answer_question"]

_UNAVAILABLE = (
    "## The figures are unavailable right now\n\n"
    "Please try that question again in a moment."
)
_OFFICE_KINDS = {
    "share",
    "office_total",
    "threshold",
    "average",
    "extreme",
    "top",
    "rank",
    "list",
}


def answer_question(question: str, prior: list[str], run_sql: SqlRunner) -> FigureResult:
    """Look up the question. ``prior`` is earlier user turns in the same chat.

    Status is ``skip`` for a greeting, ``answered`` or ``clarification`` when
    the reply is ready, and ``scope`` when the office and period are known but
    the wording still needs the language model to finish the answer.
    """
    current = (question or "").strip()
    if not current or _GREETING.fullmatch(current):
        return FigureResult(status="skip")

    history = [turn.strip() for turn in prior if turn and turn.strip()]
    combined = "\n".join([*history, current])
    catalog = load_catalog()
    cards = relevant_tables(catalog, combined)
    card = _fact_card(cards)
    steps: list[dict] = []
    if card is None:
        steps.append(_step("catalogue", "Checked the catalogue", "No figure table matched this question."))
        return _with_steps(
            _scope(catalog, combined, office="", grain="MTD", population="", channel=""),
            steps,
        )

    steps.append(_step("catalogue", "Matched the catalogue", _catalogue_detail(card)))
    try:
        offices, month_id, people, designations, group_values = _lookups(card, run_sql)
    except Exception:
        logger.exception("Figure lookup failed")
        steps.append(_step("lookups", "Loaded the live lists", "The lookup did not finish."))
        return _with_steps(_unavailable(), steps)

    steps.append(_step("lookups", "Loaded the live lists", _lookup_detail(card)))
    office, ambiguous_offices = _bind_office(combined, offices)
    steps.append(_step("scope", "Resolved the scope", _scope_detail(office, ambiguous_offices, month_id)))
    hierarchy = _hierarchy_card(catalog, combined)
    kwargs = {
        "people": tuple(people),
        "designations": tuple(designations),
        "group_values": group_values,
        "office": office,
        "hierarchy": hierarchy,
        "phrases": catalog.phrases,
    }
    # A complete question stands on its own. Earlier turns are only for a
    # fragment such as "for direct agency" or "what about this month?".
    reading = interpret(current, card, **kwargs)
    if not reading.kind():
        reading = _metrics_for_turn(interpret(combined, card, **kwargs), current)
    steps.append(_step("reading", "Read the question", _reading_detail(reading)))
    if ambiguous_offices and (
        reading.kind() in _OFFICE_KINDS or _needs_office(combined, catalog.phrases)
    ):
        office_question = _office_question(ambiguous_offices)
        steps.append(_step("follow-up", "Asked a follow-up", office_question))
        return _with_steps(
            FigureResult(status="clarification", markdown=office_question, brief=office_question),
            steps,
        )

    if reading.kind():
        try:
            answered = execute(
                reading,
                card=card,
                hierarchy=hierarchy,
                office=office,
                named_office=bool(office) and normalize(office) in normalize(combined),
                month_id=month_id,
                run_sql=run_sql,
            )
        except Exception:
            logger.exception("Figure query failed")
            steps.append(_step("query", "Ran the query", "The query did not finish."))
            return _with_steps(_unavailable(), steps)
        if answered is not None:
            if answered.sql.strip():
                steps.append(_step("query", "Ran the query", textwrap.dedent(answered.sql).strip()))
            if answered.support_sql.strip():
                steps.append(
                    _step("count", "Counted the group", textwrap.dedent(answered.support_sql).strip())
                )
            noun = "row" if answered.row_count == 1 else "rows"
            steps.append(_step("rows", "Formatted the answer", f"{answered.row_count} {noun}."))
            return _with_steps(answered, steps)

    if len(reading.named_people) > 1 and not reading.kind():
        names = name_list(list(reading.named_people))
        question_text = f"Which person should I use? {names}."
        steps.append(_step("follow-up", "Asked a follow-up", question_text))
        return _with_steps(
            FigureResult(status="clarification", markdown=question_text, brief=question_text),
            steps,
        )

    if not reading.metrics and not reading.kind() and not history:
        label = _spoken_population(reading, normalize(current))
        question_text = f"What would you like to know about {label}?"
        steps.append(_step("follow-up", "Asked a follow-up", question_text))
        return _with_steps(
            FigureResult(status="clarification", markdown=question_text, brief=question_text),
            steps,
        )

    steps.append(
        _step("handoff", "Passed the question on", "The catalogue did not have a finished answer for this wording.")
    )
    return _with_steps(
        _scope(
            catalog,
            combined,
            office=office,
            grain=reading.grain,
            population=_spoken_population(reading, normalize(combined)) if reading.population else "",
            channel=reading.channel[1] if reading.channel else "",
        ),
        steps,
    )


def _step(step_id: str, title: str, detail: str) -> dict:
    return {
        "id": step_id,
        "kind": "calculation",
        "title": title,
        "detail": (detail or "").strip(),
        "status": "done",
    }


def _with_steps(result: FigureResult, steps: list[dict]) -> FigureResult:
    """Keep the answer and attach the stages that produced it."""
    kept = tuple(step for step in steps if step.get("title"))
    if not kept:
        return result
    return replace(result, steps=kept)


def _catalogue_detail(card: TableCard) -> str:
    lines = [card.name, card.table]
    if card.summary:
        lines.append(card.summary)
    return "\n".join(lines)


def _lookup_detail(card: TableCard) -> str:
    blocks: list[str] = []
    names = ["offices", "people", "designations", "latest_month"]
    names.extend(group.lookup for group in card.groups if group.lookup)
    seen: set[str] = set()
    for name in names:
        sql = (card.lookups.get(name) or "").strip()
        if not name or name in seen or not sql:
            continue
        seen.add(name)
        blocks.append(f"{name}\n{sql}")
    return "\n\n".join(blocks)


def _scope_detail(office: str, ambiguous: list[str], month_id: str) -> str:
    if ambiguous:
        office_line = "Office: not chosen\nCandidates: " + ", ".join(ambiguous)
    else:
        office_line = f"Office: {office or 'not set'}"
    return f"{office_line}\nLatest month: {month_id or 'unknown'}"


def _reading_detail(reading: Reading) -> str:
    lines = [
        f"Shape: {reading.kind() or 'not answered from the catalogue'}",
        f"Period: {_GRAIN_LABEL.get(reading.grain, reading.grain)}",
    ]
    if reading.metrics:
        lines.append("Measures: " + ", ".join(metric.label for metric in reading.metrics))
    if reading.population is not None:
        lines.append(f"Population: {reading.population.phrases[0]}")
    if reading.comparisons:
        lines.append(
            "Cutoffs: " + "; ".join(f"{item.metric.label} {item.label.strip()}" for item in reading.comparisons)
        )
    if reading.person:
        lines.append(f"Person: {reading.person}")
    if reading.named_people and not reading.person:
        lines.append("People: " + ", ".join(reading.named_people))
    if reading.designation:
        lines.append(f"Designation: {reading.designation}")
    if reading.group is not None:
        values = ", ".join(reading.group_values)
        lines.append(f"Group: {reading.group.label}" + (f" ({values})" if values else ""))
    if reading.top_n:
        lines.append(f"Limit: top {reading.top_n}")
    if reading.extreme:
        lines.append(f"Extreme: {reading.extreme}")
    if reading.roles:
        lines.append("Roles: " + ", ".join(role.label for role in reading.roles))
    return "\n".join(lines)


def _metrics_for_turn(reading: Reading, current: str) -> Reading:
    """Keep measures named in this turn. Earlier measures stay when this turn names none."""
    text = normalize(current)
    chosen = tuple(
        metric
        for metric in reading.metrics
        if any(contains_phrase(text, normalize(phrase)) for phrase in metric.phrases)
    )
    if not chosen or len(chosen) == len(reading.metrics):
        return reading
    chosen_ids = {metric.id for metric in chosen}
    comparisons = tuple(item for item in reading.comparisons if item.metric.id in chosen_ids)
    return replace(reading, metrics=chosen, comparisons=comparisons)


def _spoken_population(reading: Reading, text: str) -> str:
    if reading.population is None:
        return "the figures"
    best = reading.population.phrases[0]
    best_len = -1
    for phrase in reading.population.phrases:
        token = normalize(phrase)
        if token and contains_phrase(text, token) and len(token) > best_len:
            best = phrase
            best_len = len(token)
    return best


def _unavailable() -> FigureResult:
    return FigureResult(status="answered", markdown=_UNAVAILABLE, brief=_UNAVAILABLE)


def _scope(
    catalog: Catalog,
    question: str,
    office: str,
    grain: str,
    population: str,
    channel: str,
) -> FigureResult:
    docs = relevant_documents(catalog, question)
    lines = [
        "status: scope",
        f"office: {office or 'not required'}",
        f"period: {_GRAIN_LABEL.get(grain, grain)}",
        f"population: {population or 'not set'}",
        f"channel: {channel or 'not set'}",
        "A ratio is stored between 0 and 1. A percent is already in percent points.",
        "Ask a follow-up only when several offices match and the question does not name one.",
        "A phrase such as direct agency is a group of people, not an office name.",
    ]
    for doc in docs:
        lines.append(f"document: {doc.title}. {doc.summary}")
    return FigureResult(status="scope", brief="\n".join(lines))


def _fact_card(cards: tuple[TableCard, ...]) -> TableCard | None:
    for card in cards:
        if card.lookups.get("offices") and card.metrics:
            return card
    for card in cards:
        if card.metrics:
            return card
    return cards[0] if cards else None


def _hierarchy_card(catalog: Catalog, question: str) -> TableCard | None:
    text = normalize(question)
    for card in catalog.tables:
        if any(contains_phrase(text, normalize(phrase)) for role in card.roles for phrase in role.phrases):
            return card
    return None


def _lookups(
    card: TableCard,
    run_sql: SqlRunner,
) -> tuple[list[str], str, list[str], list[str], dict[str, tuple[str, ...]]]:
    offices = _column(card, run_sql, "offices")
    people = _column(card, run_sql, "people")
    designations = _column(card, run_sql, "designations")
    month_id = ""
    if card.lookups.get("latest_month"):
        _, rows = run_sql(card.lookups["latest_month"])
        if rows and rows[0] and rows[0][0]:
            month_id = str(rows[0][0]).split(".", maxsplit=1)[0]
    groups: dict[str, tuple[str, ...]] = {}
    for group in card.groups:
        if group.lookup and group.lookup in card.lookups:
            groups[group.lookup] = tuple(_column(card, run_sql, group.lookup))
    return offices, month_id, people, designations, groups


def _column(card: TableCard, run_sql: SqlRunner, name: str) -> list[str]:
    sql = card.lookups.get(name)
    if not sql:
        return []
    _, rows = run_sql(sql)
    return [row[0].strip() for row in rows if row and row[0] and row[0].strip()]


def _bind_office(question: str, offices: list[str]) -> tuple[str, list[str]]:
    """Return the office to use, or the offices the person still has to choose.

    One office is used without asking. Several offices ask only when the
    question would otherwise mix them. A named person does not need an office.
    """
    lowered = normalize(question)
    named = [office for office in offices if contains_phrase(lowered, normalize(office))]
    if len(named) == 1:
        return named[0], []
    if len(named) > 1:
        return "", named
    if len(offices) == 1:
        return offices[0], []
    if len(offices) > 1:
        return "", list(offices)
    return "", []


def _office_question(offices: list[str]) -> str:
    return f"Which office should I use? The offices in the figures are {name_list(offices)}."


def _needs_office(question: str, phrases: tuple[PhraseBinding, ...]) -> bool:
    text = normalize(question)
    if re.search(r"\b(share|top\s+\d+)\b", text):
        return True
    return any(
        binding.bind == "office"
        and any(contains_phrase(text, normalize(phrase)) for phrase in binding.phrases)
        for binding in phrases
    )
