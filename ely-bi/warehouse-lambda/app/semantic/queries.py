"""Run one catalog reading as a read-only query and a business answer.

Identifiers come from the catalog. Values that came from the warehouse are
escaped. A cutoff on a ratio is already in ratio units, so 90% compares with 0.90.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from app.semantic.catalog import Filter, Metric, Population, TableCard
from app.semantic.format import format_measure, format_number, markdown_table, name_list, parse_decimal
from app.semantic.interpret import Comparison, Reading

SqlRunner = Callable[[str], tuple[list[str], list[list[str]]]]


@dataclass(frozen=True)
class FigureResult:
    """What the assistant should do after looking at the figures."""

    status: str
    markdown: str = ""
    brief: str = ""
    assumption: str = ""
    sql: str = ""
    row_count: int = 0
    support_sql: str = ""
    steps: tuple[dict, ...] = ()

_GRAIN = {
    "MTD": "month to date",
    "QTD": "quarter to date",
    "YTD": "year to date",
    "FTD": "for the day",
}
_OPS = {">", "<", ">=", "<=", "="}


def execute(
    reading: Reading,
    *,
    card: TableCard,
    hierarchy: TableCard | None,
    office: str,
    named_office: bool,
    month_id: str,
    run_sql: SqlRunner,
) -> FigureResult | None:
    """Answer a reading. ``None`` means this shape is not supported yet."""
    kind = reading.kind()
    if kind == "hierarchy" and hierarchy is not None:
        return _hierarchy(reading, hierarchy, office, run_sql)
    if not month_id or not kind:
        return None
    handlers = {
        "share": _share,
        "office_total": _office_total,
        "average": _average,
        "threshold": _threshold,
        "extreme": _extreme,
        "top": _top,
        "rank": _rank,
        "person": _person,
        "people": _people,
        "list": _list,
    }
    handler = handlers.get(kind)
    if handler is None:
        return None
    return handler(reading, card, office, named_office, month_id, run_sql)


def _share(reading, card, office, named_office, month_id, run_sql) -> FigureResult:
    metric = reading.metrics[0]
    population = reading.population or _population_by_id(card, "direct_reports")
    if population is None:
        raise ValueError("No producer population in the catalog")
    top_n = reading.top_n or 3
    people_where = _where(
        card, reading, month_id, office, population.filters, member=True, exclude_rollup=True
    )
    office_where = _where(
        card, reading, month_id, office, card.office_head_filters, member=False, name=office, name_column=card.office_head_column
    )
    sql = f"""
        WITH office_total AS (
            SELECT MAX({metric.column}) AS office_value
            FROM {card.table}
            WHERE {office_where}
        ),
        ranked AS (
            SELECT
                {card.office_head_column} AS person_name,
                {metric.column} AS metric_value,
                ROW_NUMBER() OVER (
                ORDER BY {metric.column} DESC NULLS LAST, {card.office_head_column}
            ) AS place
            FROM {card.table}
            WHERE {people_where}
        )
        SELECT ranked.place, ranked.person_name, ranked.metric_value, office_total.office_value
        FROM ranked
        CROSS JOIN office_total
        WHERE ranked.place <= {int(top_n)}
        ORDER BY ranked.place
    """
    _, rows = run_sql(sql)
    return _share_markdown(metric, rows, office, reading, named_office, top_n, sql)


def _office_total(reading, card, office, named_office, month_id, run_sql) -> FigureResult:
    metrics = _select_metrics(reading, card)
    where = _where(
        card,
        reading,
        month_id,
        office,
        card.office_head_filters,
        member=False,
        name=office,
        name_column=card.office_head_column,
    )
    sql = _select(card, metrics, where, limit=5)
    _, rows = run_sql(sql)
    if not rows:
        return _empty(office, reading, sql)
    values = [
        f"{format_measure(_required(rows[0][index + 1]), metric.scale)} {metric.label}"
        for index, metric in enumerate(metrics)
    ]
    assumption = _assumption(office, reading, named_office)
    sentence = f"{office}'s office total is {' and '.join(values)}."
    markdown = f"## {sentence}\n\n{assumption}"
    return _done(markdown, assumption, sql, len(rows))


def _average(reading, card, office, named_office, month_id, run_sql) -> FigureResult:
    metric = reading.metrics[0]
    group = reading.group
    where = _where(
        card, reading, month_id, office, _member_filters(reading, card), member=True, exclude_rollup=True
    )
    where = f"{where} AND {metric.column} IS NOT NULL"
    if group is None:
        sql = f"SELECT AVG({metric.column}) AS average_value, COUNT(*) AS group_count FROM {card.table} WHERE {where}"
        _, rows = run_sql(sql)
        count = parse_decimal(rows[0][1]) if rows else None
        if not rows or not count:
            return _empty(office, reading, sql)
        shown = format_measure(_required(rows[0][0]), metric.scale)
        noun = _count_noun(reading, None, int(count))
        assumption = _assumption(office, reading, named_office)
        sentence = f"Average {metric.label} is {shown} across {format_number(count)} {noun}."
        markdown = f"## {sentence}\n\n{assumption}"
        return _done(markdown, assumption, sql, 1)
    value_sql = ""
    if reading.group_values:
        values = ", ".join(_literal(value) for value in reading.group_values)
        value_sql = f" AND {group.column} IN ({values})"
    sql = f"""
        SELECT {group.column} AS group_name, AVG({metric.column}) AS average_value, COUNT(*) AS group_count
        FROM {card.table}
        WHERE {where}
          {value_sql}
          AND {group.column} IS NOT NULL
        GROUP BY {group.column}
    """
    _, rows = run_sql(sql)
    order = {normalize_key(value): index for index, value in enumerate(reading.group_values)}
    rows.sort(key=lambda row: (order.get(normalize_key(row[0]), 99), normalize_key(row[0])))
    if not rows:
        return _empty(office, reading, sql)
    lines = []
    body = []
    for row in rows:
        average = parse_decimal(row[1]) or Decimal("0")
        count = parse_decimal(row[2]) or Decimal("0")
        shown = format_measure(average, metric.scale)
        noun = _count_noun(reading, group, int(count))
        lines.append(f"{row[0]} averages {shown} across {format_number(count)} {noun}")
        body.append((row[0], shown, format_number(count)))
    assumption = _assumption(office, reading, named_office)
    markdown = (
        f"## {metric.label} by {group.label}\n\n{assumption}\n\n"
        + ". ".join(lines)
        + ".\n\n"
        + markdown_table(["Group", metric.label, "People"], body)
    )
    return _done(markdown, assumption, sql, len(rows))


def _threshold(reading, card, office, named_office, month_id, run_sql) -> FigureResult:
    metrics = _select_metrics(reading, card)
    where = _where(
        card,
        reading,
        month_id,
        office,
        _member_filters(reading, card),
        member=True,
        exclude_rollup=True,
        comparisons=reading.comparisons,
    )
    sql = _select(card, metrics, where, group=reading.group, limit=100)
    _, rows = run_sql(sql)
    total, population_sql = _count(card, reading, month_id, office, run_sql)
    rows = _sort_thresholds(rows, metrics, reading.comparisons)
    assumption = _assumption(office, reading, named_office)
    if not rows:
        label = _population_label(reading)
        markdown = f"## No {label} match those cutoffs\n\n{assumption}\n\nNobody in that group meets every cutoff."
        return _done(markdown, assumption, sql, 0, support_sql=population_sql)
    group_offset = 1 + len(metrics)
    body = []
    for row in rows:
        values = [
            format_measure(_required(row[index + 1]), metric.scale) for index, metric in enumerate(metrics)
        ]
        if reading.group is not None and len(row) > group_offset and row[group_offset]:
            values.append(row[group_offset])
        body.append((row[0], *values))
    headers = ["Person", *[metric.label for metric in metrics]]
    if reading.group is not None:
        headers.append("Track")
    matched = len(rows)
    group_label = _population_label(reading, matched)
    cutoff = " and ".join(
        f"{_sentence_label(item.metric.label)} {item.label.strip()}" for item in reading.comparisons
    )
    if reading.population and total:
        subject = f"{matched} of {total} {group_label}"
    else:
        subject = f"{matched} {group_label}"
    verb = "has" if matched == 1 else "have"
    sentence = f"{subject} {verb} {cutoff}."
    direction = ""
    if matched > 1 and any(item.op.startswith("<") for item in reading.comparisons):
        direction = "Lowest first."
    blocks = [f"## {sentence}", assumption]
    if direction:
        blocks.append(direction)
    blocks.append(markdown_table(headers, body))
    markdown = "\n\n".join(blocks)
    return _done(markdown, assumption, sql, matched, support_sql=population_sql)


def _extreme(reading, card, office, named_office, month_id, run_sql) -> FigureResult:
    metrics = _select_metrics(reading, card)
    primary = metrics[0]
    where = _where(
        card, reading, month_id, office, _member_filters(reading, card), member=True, exclude_rollup=True
    )
    where = f"{where} AND {primary.column} IS NOT NULL"
    direction = "DESC" if reading.extreme == "highest" else "ASC"
    sql = _select(
        card,
        metrics,
        where,
        order=f"{primary.column} {direction} NULLS LAST, {card.office_head_column}",
        limit=1,
    )
    _, rows = run_sql(sql)
    if not rows:
        return _empty(office, reading, sql)
    row = rows[0]
    shown = format_measure(_required(row[1]), primary.scale)
    extra = ""
    if len(metrics) > 1 and row[2] not in ("", None):
        extra = f", with {format_measure(_required(row[2]), metrics[1].scale)} {metrics[1].label.lower()}"
    assumption = _assumption(office, reading, named_office)
    sentence = f"{row[0]} has the {reading.extreme} {primary.label} {_period(reading)}, at {shown}{extra}."
    headers = ["Person", *[metric.label for metric in metrics]]
    body = [(row[0], *[format_measure(_required(row[index + 1]), metric.scale) for index, metric in enumerate(metrics)])]
    markdown = f"## {sentence}\n\n{assumption}\n\n{markdown_table(headers, body)}"
    return _done(markdown, assumption, sql, 1)


def _top(reading, card, office, named_office, month_id, run_sql) -> FigureResult:
    metrics = _select_metrics(reading, card)
    where = _where(
        card, reading, month_id, office, _member_filters(reading, card), member=True, exclude_rollup=True
    )
    sql = _select(
        card,
        metrics,
        where,
        order=f"{metrics[0].column} DESC NULLS LAST, {card.office_head_column}",
        limit=int(reading.top_n or 5),
    )
    _, rows = run_sql(sql)
    return _ranked_markdown(metrics, rows, office, reading, named_office, int(reading.top_n or 5), sql, "Top")


def _rank(reading, card, office, named_office, month_id, run_sql) -> FigureResult:
    metrics = _select_metrics(reading, card)
    filters = _member_filters(reading, card)
    where = _where(card, reading, month_id, office, filters, member=True, exclude_rollup=True, designation=reading.designation)
    sql = _select(
        card,
        metrics,
        where,
        order=f"{metrics[0].column} DESC NULLS LAST, {card.office_head_column}",
        limit=25,
    )
    _, rows = run_sql(sql)
    title = reading.designation or "People"
    return _ranked_markdown(metrics, rows, office, reading, named_office, len(rows), sql, title)


def _person(reading, card, office, named_office, month_id, run_sql) -> FigureResult:
    metrics = _select_metrics(reading, card)
    where = _where(
        card,
        reading,
        month_id,
        office,
        (),
        member=False,
        name=reading.person,
        name_column=card.office_head_column,
    )
    columns = ", ".join(f"MAX({metric.column}) AS {metric.column}" for metric in metrics)
    if reading.group is not None and "promotion" in reading.group.column:
        columns = f"MAX({reading.group.column}) AS track, {columns}"
    sql = f"""
        SELECT {card.office_head_column} AS person_name, {columns}
        FROM {card.table}
        WHERE {where}
        GROUP BY {card.office_head_column}
        LIMIT 5
    """
    _, rows = run_sql(sql)
    if not rows:
        return _empty(office, reading, sql)
    return _person_markdown(reading, rows[0], metrics, office, named_office, sql)


def _people(reading, card, office, named_office, month_id, run_sql) -> FigureResult:
    metrics = _select_metrics(reading, card)
    where = _where(card, reading, month_id, office, (), member=False)
    names = ", ".join(_literal(name) for name in reading.named_people)
    columns = ", ".join(f"MAX({metric.column}) AS {metric.column}" for metric in metrics)
    sql = f"""
        SELECT {card.office_head_column} AS person_name, {columns}
        FROM {card.table}
        WHERE {where}
          AND {card.office_head_column} IN ({names})
        GROUP BY {card.office_head_column}
        LIMIT 25
    """
    _, rows = run_sql(sql)
    if not rows:
        return _empty(office, reading, sql)
    order = {name.lower(): index for index, name in enumerate(reading.named_people)}
    rows.sort(key=lambda row: order.get((row[0] or "").lower(), 99))
    assumption = _assumption(office, reading, named_office)
    sentences = []
    body = []
    for row in rows:
        parts = [
            _measure_phrase(metric, row[index + 1]) for index, metric in enumerate(metrics)
        ]
        sentences.append(f"{row[0]} has {' and '.join(parts)}")
        body.append(
            (
                row[0],
                *[
                    format_measure(_required(row[index + 1]), metric.scale)
                    for index, metric in enumerate(metrics)
                ],
            )
        )
    headers = ["Person", *[metric.label for metric in metrics]]
    markdown = (
        f"## {'. '.join(sentences)}.\n\n{assumption}\n\n{markdown_table(headers, body)}"
    )
    return _done(markdown, assumption, sql, len(rows))


def _person_markdown(reading, row, metrics, office, named_office, sql) -> FigureResult:
    offset = 1
    track = ""
    if reading.group is not None and "promotion" in reading.group.column:
        track = row[1]
        offset = 2
    parts = [_measure_phrase(metric, row[offset + index]) for index, metric in enumerate(metrics)]
    assumption = _assumption(office, reading, named_office)
    sentence = f"{reading.person} has {' and '.join(parts)} {_period(reading)}."
    if track:
        sentence = f"{reading.person}, on a {track} promotion track, has {' and '.join(parts)}."
    markdown = f"## {sentence}\n\n{assumption}"
    return _done(markdown, assumption, sql, 1)


def _list(reading, card, office, named_office, month_id, run_sql) -> FigureResult:
    filters = _member_filters(reading, card)
    where = _where(
        card,
        reading,
        month_id,
        office,
        filters,
        member=True,
        exclude_rollup=True,
        designation=reading.designation,
    )
    sql = f"""
        SELECT DISTINCT {card.office_head_column} AS person_name
        FROM {card.table}
        WHERE {where}
        ORDER BY person_name
        LIMIT 100
    """
    _, rows = run_sql(sql)
    names = [row[0] for row in rows if row and row[0]]
    assumption = _assumption(office, reading, named_office)
    label = reading.designation or _population_label(reading)
    if not names:
        markdown = f"## No {label} for {office}\n\n{assumption}"
        return _done(markdown, assumption, sql, 0)
    sentence = f"{len(names)} people: {name_list(names)}."
    markdown = f"## {label} reporting to {office}\n\n{assumption}\n\n{sentence}"
    return _done(markdown, assumption, sql, len(names))


def _hierarchy(reading, card, office, run_sql) -> FigureResult:
    person = reading.person or office
    if not person:
        return None
    columns = list(dict.fromkeys(role.name_column for role in reading.roles))
    where = _filter_sql(card.defaults)
    sql = f"""
        SELECT {", ".join(columns)}
        FROM {card.table}
        WHERE {where}
          AND {card.office_head_column} = {_literal(person)}
        LIMIT 5
    """
    _, rows = run_sql(sql)
    if not rows:
        text = f"## No reporting line for {person}"
        return _done(text, "", sql, 0)
    by_column = {
        column: rows[0][index] if index < len(rows[0]) else ""
        for index, column in enumerate(columns)
    }
    parts = []
    for role in reading.roles:
        value = by_column.get(role.name_column) or ""
        if value:
            parts.append(f"{role.label} {value}")
    if not parts:
        text = f"## No reporting line for {person}"
        return _done(text, "", sql, 0)
    sentence = ". ".join(parts) + "."
    markdown = f"## Above {person}\n\n{sentence}"
    return _done(markdown, "", sql, 1)


def _select_metrics(reading: Reading, card: TableCard) -> tuple[Metric, ...]:
    chosen: list[Metric] = list(reading.metrics)
    by_id = {metric.id: metric for metric in card.metrics}
    for metric in list(chosen):
        if reading.show_companions or reading.extreme:
            chosen.extend(metric.companions)
        for metric_id in metric.also_show:
            extra = by_id.get(metric_id)
            if extra is not None and reading.kind() in {"top", "rank", "extreme"}:
                chosen.append(extra)
    unique: list[Metric] = []
    seen: set[str] = set()
    for metric in chosen:
        if metric.column in seen:
            continue
        seen.add(metric.column)
        unique.append(metric)
    return tuple(unique)


def _member_filters(reading: Reading, card: TableCard) -> tuple[Filter, ...]:
    if reading.population is not None:
        return reading.population.filters
    if reading.kind() in {"top", "share"}:
        population = _population_by_id(card, "direct_reports")
        return population.filters if population is not None else ()
    return ()


def _where(
    card: TableCard,
    reading: Reading,
    month_id: str,
    office: str,
    filters: tuple[Filter, ...],
    *,
    member: bool,
    exclude_rollup: bool = False,
    name: str | None = None,
    name_column: str | None = None,
    comparisons: tuple[Comparison, ...] = (),
    designation: str | None = None,
) -> str:
    parts = [_filter_sql(card.defaults), f"{card.grain_column} = {_literal(reading.grain)}"]
    if month_id.isdigit():
        parts.append(f"{card.month_column} = {int(month_id)}")
    if filters:
        parts.append(_filter_sql(filters))
    if member and office and card.office_member_column:
        parts.append(f"{card.office_member_column} = {_literal(office)}")
    if name and name_column:
        parts.append(f"{name_column} = {_literal(name)}")
    if exclude_rollup and card.office_head_filters:
        parts.append("NOT (" + _filter_sql(card.office_head_filters) + ")")
    if designation and card.designation_column:
        parts.append(f"{card.designation_column} = {_literal(designation)}")
    if reading.channel:
        parts.append(f"{reading.channel[0]} = {_literal(reading.channel[1])}")
    for item in comparisons:
        if item.op not in _OPS:
            continue
        parts.append(f"{item.metric.column} IS NOT NULL")
        parts.append(f"{item.metric.column} {item.op} {_number(item.bound)}")
    return " AND ".join(part for part in parts if part)


def _select(
    card: TableCard,
    metrics: tuple[Metric, ...],
    where: str,
    *,
    order: str = "",
    limit: int = 25,
    group: GroupBy | None = None,
) -> str:
    columns = ", ".join(metric.column for metric in metrics)
    group_column = f", {group.column}" if group is not None else ""
    order_sql = f" ORDER BY {order}" if order else ""
    return f"""
        SELECT {card.office_head_column} AS person_name, {columns}{group_column}
        FROM {card.table}
        WHERE {where}
        {order_sql}
        LIMIT {int(limit)}
    """


def _count(card, reading, month_id, office, run_sql) -> tuple[int, str]:
    """Return the size of the group, and the query that counted it."""
    where = _where(
        card, reading, month_id, office, _member_filters(reading, card), member=True, exclude_rollup=True
    )
    sql = f"SELECT COUNT(*) FROM {card.table} WHERE {where}"
    _, rows = run_sql(sql)
    if not rows or not rows[0] or not rows[0][0]:
        return 0, sql
    parsed = parse_decimal(rows[0][0])
    return int(parsed or 0), sql


def _sort_thresholds(rows, metrics, comparisons):
    """Lowest-first on a below cutoff. Ties keep alphabetical order."""
    below = next((item for item in comparisons if item.op.startswith("<")), None)
    above = next((item for item in comparisons if item.op.startswith(">")), None)
    primary = below or above
    reverse = below is None and above is not None

    def key(row: list[str]) -> tuple:
        amount = _metric_sort(row, metrics, primary.metric) if primary is not None else Decimal("0")
        if reverse:
            amount = -amount
        return (amount, row[0].lower())

    return sorted(rows, key=key)


def _metric_sort(row, metrics, metric: Metric) -> Decimal:
    for index, candidate in enumerate(metrics):
        if candidate.column == metric.column:
            return parse_decimal(row[index + 1]) or Decimal("0")
    return Decimal("0")


def _share_markdown(metric, rows, office, reading, named_office, top_n, sql) -> FigureResult:
    if not rows:
        return _empty(office, reading, sql)
    people = [(row[1], parse_decimal(row[2]) or Decimal("0")) for row in rows]
    office_value = parse_decimal(rows[0][3]) or Decimal("0")
    total = sum((amount for _, amount in people), Decimal("0"))
    share = (total / office_value * Decimal("100")).quantize(Decimal("0.1")) if office_value else Decimal("0.0")
    assumption = _assumption(office, reading, named_office)
    sentence = (
        f"The top {top_n} ({name_list([name for name, _ in people])}) produced {format_number(total)} "
        f"of {metric.label}, which is {share}% of the office total of {format_number(office_value)}."
    )
    table = markdown_table(
        ["Producer", metric.label],
        [(name, format_number(amount)) for name, amount in people],
    )
    markdown = f"## Top {top_n} producers are {share}% of office {metric.label}\n\n{assumption}\n\n{sentence}\n\n{table}"
    return _done(markdown, assumption, sql, len(rows))


def _ranked_markdown(metrics, rows, office, reading, named_office, count, sql, title) -> FigureResult:
    if not rows:
        return _empty(office, reading, sql)
    headers = ["Person", *[metric.label for metric in metrics]]
    body = []
    for row in rows:
        values = [
            format_measure(_required(row[index + 1]), metric.scale) if row[index + 1] else ""
            for index, metric in enumerate(metrics)
        ]
        body.append((row[0], *values))
    assumption = _assumption(office, reading, named_office)
    markdown = (
        f"## {title} by {metrics[0].label}\n\n{assumption}\n\n"
        f"{len(rows)} people, highest first.\n\n{markdown_table(headers, body)}"
    )
    return _done(markdown, assumption, sql, len(rows))


def _assumption(office: str, reading: Reading, named_office: bool) -> str:
    period = _period(reading)
    if not office:
        return f"This is {period}."
    if named_office:
        sentence = f"This is {period} for {office}'s office."
    else:
        sentence = f"This is {period} for {office}'s office, the only office in the figures."
    if reading.channel:
        sentence += " Direct agency is the direct producers in that office."
    return sentence


def _period(reading: Reading) -> str:
    return _GRAIN.get(reading.grain, reading.grain)


def _measure_phrase(metric: Metric, raw: str) -> str:
    shown = format_measure(_required(raw), metric.scale)
    if metric.label.lower().endswith(" size"):
        return f"a {metric.label.lower()} of {shown}"
    return f"{shown} {metric.label}"


def _sentence_label(label: str) -> str:
    if not label:
        return label
    return label[0].lower() + label[1:]


def _count_noun(reading: Reading, group, count: int | None = None) -> str:
    if group is not None and getattr(group, "count_label", ""):
        word = group.count_label
        if count == 1 and word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word
    return _population_label(reading, count)


def _population_label(reading: Reading, count: int | None = None) -> str:
    if reading.population is None:
        word = "person" if count == 1 else "people"
        return word
    if reading.population.id == "direct_reports":
        return "direct producer" if count == 1 else "direct producers"
    word = reading.population.phrases[0]
    if count == 1 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    if count not in (None, 1) and not word.endswith("s"):
        return f"{word}s"
    return word


def _population_by_id(card: TableCard, population_id: str) -> Population | None:
    for population in card.populations:
        if population.id == population_id:
            return population
    return None


def _empty(office: str, reading: Reading, sql: str) -> FigureResult:
    assumption = _assumption(office, reading, False)
    text = f"## No matching figures {_period(reading)}\n\n{assumption}\n\nNothing in the figures matches that question."
    return _done(text, assumption, sql, 0)


def _done(
    markdown: str,
    assumption: str,
    sql: str,
    row_count: int,
    support_sql: str = "",
) -> FigureResult:
    cleaned = markdown.strip()
    return FigureResult(
        status="answered",
        markdown=cleaned,
        brief=cleaned,
        assumption=assumption,
        sql=sql,
        row_count=row_count,
        support_sql=support_sql,
    )


def _required(value: str) -> Decimal:
    return parse_decimal(value) or Decimal("0")


def _filter_sql(filters: tuple[Filter, ...]) -> str:
    if not filters:
        return ""
    return " AND ".join(f"{item.column} = {_literal(item.equals)}" for item in filters)


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _number(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def normalize_key(value: str) -> str:
    return (value or "").replace("-", " ").replace(".", " ").lower().strip()
