"""Load the semantic catalog from ``data/semantic``.

Each table, phrase, and document is a small JSON file. Nothing here is hardcoded
to a fixed list of columns beyond what those files declare.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.semantic.format import tokens


_IDENTIFIER = r"^[A-Za-z_][A-Za-z0-9_]*$"
_SELECT = ("select", "with")
_FORBIDDEN = ("insert ", "update ", "delete ", "drop ", "alter ", "create ", "grant ", "truncate ")


@dataclass(frozen=True)
class Filter:
    """One equality filter declared in a catalog file."""

    column: str
    equals: str


@dataclass(frozen=True)
class Metric:
    """A business measure and the column that stores it.

    ``scale`` is ``number``, ``ratio`` (0.248 means 24.8%), or ``percent``
    (50 means 50%). ``companions`` are figures that belong with this measure,
    such as the count behind a rate. ``also_show`` names other metric ids to
    include when this measure is ranked.
    """

    id: str
    phrases: tuple[str, ...]
    column: str
    scale: str
    label: str
    table: str
    companions: tuple["Metric", ...] = ()
    also_show: tuple[str, ...] = ()


@dataclass(frozen=True)
class Population:
    """A set of rows a person might mean, such as producers or the office total."""

    id: str
    phrases: tuple[str, ...]
    filters: tuple[Filter, ...]
    table: str


@dataclass(frozen=True)
class GroupBy:
    """A column whose distinct values can be compared, such as promotion track."""

    id: str
    column: str
    lookup: str
    phrases: tuple[str, ...]
    label: str
    count_label: str = "people"


@dataclass(frozen=True)
class RoleField:
    """A named role above a person, such as RM or ZVP."""

    id: str
    phrases: tuple[str, ...]
    name_column: str
    label: str


@dataclass(frozen=True)
class TableCard:
    """Retrieval card for one physical table."""

    table: str
    schema: str
    name: str
    summary: str
    help_text: str
    defaults: tuple[Filter, ...]
    metrics: tuple[Metric, ...]
    populations: tuple[Population, ...]
    dimensions: tuple[tuple[str, str], ...]
    grains: tuple[tuple[tuple[str, ...], str], ...]
    default_grain: str
    month_column: str
    grain_column: str
    office_head_column: str
    office_member_column: str
    office_head_filters: tuple[Filter, ...]
    channels: tuple[tuple[tuple[str, ...], str, str], ...]
    lookups: dict[str, str]
    groups: tuple[GroupBy, ...] = ()
    roles: tuple[RoleField, ...] = ()
    designation_column: str = ""


@dataclass(frozen=True)
class PhraseBinding:
    """A business phrase that is not itself a stored name."""

    id: str
    phrases: tuple[str, ...]
    population: str
    bind: str
    note: str


@dataclass(frozen=True)
class DocumentCard:
    """A short card for one S3 document. The body stays in S3."""

    id: str
    title: str
    topics: tuple[str, ...]
    summary: str
    s3_key: str


@dataclass(frozen=True)
class Catalog:
    """Everything the resolver is allowed to know about the figures."""

    tables: tuple[TableCard, ...]
    phrases: tuple[PhraseBinding, ...]
    documents: tuple[DocumentCard, ...]


def catalog_dir() -> Path:
    """Directory of catalog JSON. Override with SEMANTIC_CATALOG_DIR in Lambda."""
    override = os.environ.get("SEMANTIC_CATALOG_DIR", "").strip()
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data" / "semantic"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("data/semantic catalog directory was not found")


def load_catalog() -> Catalog:
    """Load the catalog, cached until a JSON file changes."""
    root = catalog_dir()
    stamp = _stamp(root)
    return _load_cached(str(root), stamp)


def _stamp(root: Path) -> str:
    parts: list[str] = []
    for path in sorted(root.rglob("*.json")):
        parts.append(f"{path.name}:{path.stat().st_mtime_ns}")
    return "|".join(parts)


@lru_cache(maxsize=4)
def _load_cached(root: str, stamp: str) -> Catalog:  # noqa: ARG001
    base = Path(root)
    tables = tuple(
        _table_card(path)
        for path in sorted((base / "tables").glob("*.json"))
    )
    phrases = _phrases(base / "phrases.json")
    documents = tuple(
        _document_card(path)
        for path in sorted((base / "documents").glob("*.json"))
    ) if (base / "documents").is_dir() else ()
    return Catalog(tables=tables, phrases=phrases, documents=documents)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return payload


def _table_card(path: Path) -> TableCard:
    raw = _read(path)
    qualified = str(raw.get("table") or "").strip()
    schema, _, name = qualified.partition(".")
    if not schema or not name:
        raise ValueError(f"{path.name} needs a schema.table name")
    _ident(schema, path.name)
    _ident(name, path.name)
    time_raw = raw.get("time") or {}
    office_raw = raw.get("office") or {}
    metrics = tuple(_metric(item, qualified, path.name) for item in raw.get("metrics") or [])
    populations = tuple(
        Population(
            id=str(item["id"]),
            phrases=_phrases_of(item),
            filters=_filters(item.get("filters") or [], path.name),
            table=qualified,
        )
        for item in raw.get("populations") or []
    )
    grains = tuple(
        (_phrases_of(item), str(item["value"]).strip())
        for item in time_raw.get("grains") or []
    )
    channels = tuple(
        (_phrases_of(item), _ident(str(item["column"]), path.name), str(item["equals"]))
        for item in raw.get("channels") or []
    )
    lookups = {
        str(key): _select_sql(str(sql), path.name)
        for key, sql in (raw.get("lookups") or {}).items()
    }
    dimensions = tuple(
        (_ident(str(item["name"]), path.name), str(item.get("means") or ""))
        for item in raw.get("dimensions") or []
    )
    groups = tuple(
        GroupBy(
            id=str(item.get("id") or item["column"]),
            column=_ident(str(item["column"]), path.name),
            lookup=str(item.get("lookup") or ""),
            phrases=_optional_phrases(item),
            label=str(item.get("label") or item.get("id") or item["column"]),
            count_label=str(item.get("count_label") or "people"),
        )
        for item in raw.get("groups") or []
    )
    roles = tuple(
        RoleField(
            id=str(item.get("id") or item["label"]),
            phrases=_phrases_of(item),
            name_column=_ident(str(item["name_column"]), path.name),
            label=str(item.get("label") or item["id"]),
        )
        for item in raw.get("roles") or []
    )
    designation = str(raw.get("designation_column") or "").strip()
    if designation:
        _ident(designation, path.name)
    return TableCard(
        table=qualified,
        schema=schema,
        name=name,
        summary=str(raw.get("summary") or ""),
        help_text=str(raw.get("help") or ""),
        defaults=_filters(raw.get("defaults") or [], path.name),
        metrics=metrics,
        populations=populations,
        dimensions=dimensions,
        grains=grains,
        default_grain=str(time_raw.get("default_grain") or "MTD"),
        month_column=_ident(str(time_raw.get("month_column") or "month_id"), path.name),
        grain_column=_ident(str(time_raw.get("grain_column") or "time_dim"), path.name),
        office_head_column=_ident(str(office_raw.get("head_name_column") or "agent_name"), path.name),
        office_member_column=_ident(
            str(office_raw.get("member_name_column") or "supervisor_name"),
            path.name,
        ),
        office_head_filters=_filters(office_raw.get("head_filters") or [], path.name),
        channels=channels,
        lookups=lookups,
        groups=groups,
        roles=roles,
        designation_column=designation,
    )


def _metric(item: dict[str, Any], table: str, source: str) -> Metric:
    companions = tuple(
        _metric(child, table, source) for child in item.get("companions") or []
    )
    also_show = tuple(str(metric_id) for metric_id in item.get("also_show") or [])
    scale = str(item.get("scale") or "number")
    if scale not in {"number", "ratio", "percent"}:
        raise ValueError(f"Unknown scale {scale!r} in {source}")
    return Metric(
        id=str(item["id"]),
        phrases=_phrases_of(item),
        column=_ident(str(item["column"]), source),
        scale=scale,
        label=str(item.get("label") or item["id"]),
        table=table,
        companions=companions,
        also_show=also_show,
    )


def _optional_phrases(item: dict[str, Any]) -> tuple[str, ...]:
    phrases = item.get("phrases") or []
    return tuple(str(phrase).strip().lower() for phrase in phrases if str(phrase).strip())


def _phrases(path: Path) -> tuple[PhraseBinding, ...]:
    if not path.is_file():
        return ()
    raw = _read(path)
    bindings: list[PhraseBinding] = []
    for item in raw.get("bindings") or []:
        bindings.append(
            PhraseBinding(
                id=str(item.get("id") or ""),
                phrases=_phrases_of(item),
                population=str(item.get("population") or ""),
                bind=str(item.get("bind") or ""),
                note=str(item.get("note") or ""),
            )
        )
    return tuple(bindings)


def _document_card(path: Path) -> DocumentCard:
    raw = _read(path)
    topics = raw.get("topics") or []
    return DocumentCard(
        id=str(raw.get("id") or path.stem),
        title=str(raw.get("title") or path.stem),
        topics=tuple(str(topic) for topic in topics),
        summary=str(raw.get("summary") or ""),
        s3_key=str(raw.get("s3_key") or ""),
    )


def _phrases_of(item: dict[str, Any]) -> tuple[str, ...]:
    phrases = item.get("phrases") or []
    cleaned = tuple(str(phrase).strip().lower() for phrase in phrases if str(phrase).strip())
    if not cleaned:
        raise ValueError(f"A catalog entry needs phrases: {item}")
    return cleaned


def _filters(items: list[dict[str, Any]], source: str) -> tuple[Filter, ...]:
    filters: list[Filter] = []
    for item in items:
        filters.append(
            Filter(
                column=_ident(str(item["column"]), source),
                equals=str(item["equals"]),
            )
        )
    return tuple(filters)


def _ident(value: str, source: str) -> str:
    import re

    if not re.match(_IDENTIFIER, value):
        raise ValueError(f"Unsafe identifier {value!r} in {source}")
    return value


def _select_sql(sql: str, source: str) -> str:
    text = " ".join(sql.split())
    lowered = text.lower()
    if not lowered.startswith(_SELECT):
        raise ValueError(f"Lookup in {source} must be a read query")
    if any(token in f" {lowered} " for token in _FORBIDDEN):
        raise ValueError(f"Lookup in {source} must be a read query")
    if ";" in text:
        raise ValueError(f"Lookup in {source} must be a single statement")
    return text


def relevant_tables(catalog: Catalog, question: str, limit: int = 4) -> tuple[TableCard, ...]:
    """Return the tables whose cards share the most words with the question."""
    question_tokens = tokens(question)
    ranked = sorted(
        catalog.tables,
        key=lambda card: _table_score(card, question, question_tokens),
        reverse=True,
    )
    chosen = [card for card in ranked if _table_score(card, question, question_tokens) > 0]
    if not chosen:
        return ranked[:1]
    return tuple(chosen[:limit])


def relevant_documents(catalog: Catalog, question: str, limit: int = 3) -> tuple[DocumentCard, ...]:
    """Document cards to open. The document body is not loaded here."""
    question_tokens = tokens(question)
    ranked = sorted(
        catalog.documents,
        key=lambda card: len(
            question_tokens & tokens(f"{card.title} {card.summary} {' '.join(card.topics)}")
        ),
        reverse=True,
    )
    return tuple(card for card in ranked if _doc_score(card, question_tokens) > 0)[:limit]


def catalogue_hint(question: str) -> str:
    """Short column list for SQL generation. Only tables that match the question."""
    catalog = load_catalog()
    lines: list[str] = []
    for card in relevant_tables(catalog, question):
        lines.append(f"{card.table}: {card.summary}")
        for name, means in card.dimensions:
            lines.append(f"{card.table}.{name} — {means}")
        for metric in card.metrics:
            lines.append(_metric_line(card.table, metric))
            for companion in metric.companions:
                lines.append(_metric_line(card.table, companion))
    return "\n".join(lines)


def _table_score(card: TableCard, question: str, question_tokens: set[str]) -> int:
    blob = " ".join(
        [
            card.summary,
            " ".join(phrase for metric in card.metrics for phrase in metric.phrases),
            " ".join(phrase for population in card.populations for phrase in population.phrases),
            " ".join(name for name, _ in card.dimensions),
            " ".join(meaning for _, meaning in card.dimensions),
        ]
    )
    score = len(question_tokens & tokens(blob))
    lowered = question.lower()
    for metric in card.metrics:
        if any(phrase and phrase in lowered for phrase in metric.phrases):
            score += 5
    return score


def _doc_score(card: DocumentCard, question_tokens: set[str]) -> int:
    return len(question_tokens & tokens(f"{card.title} {card.summary} {' '.join(card.topics)}"))


def _metric_line(table: str, metric: Metric) -> str:
    scale = {
        "ratio": "stored as a ratio, so 0.248 means 24.8%",
        "percent": "already a percent, so 15 means 15%",
        "number": "a number",
    }.get(metric.scale, "a number")
    words = ", ".join(metric.phrases)
    return f"{table}.{metric.column} — {metric.label} ({words}); {scale}"
