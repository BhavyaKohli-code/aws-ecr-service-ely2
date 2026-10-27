"""Normalize wording and format warehouse numbers for a business reader."""

from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP

_SPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")

_ONE = Decimal("0.1")
_CENT = Decimal("0.01")


def normalize(text: str) -> str:
    """Lowercase text and treat hyphens as spaces.

    ``year-to-date`` and ``year to date`` become the same phrase.
    Decimal points and thousands separators stay so ``80.5%`` and ``1,000``
    still parse as numbers. A period that is not a decimal becomes a space,
    so ``Sr.`` matches ``Sr``.
    """
    folded = (text or "").lower().replace("–", " ").replace("—", " ").replace("-", " ")
    folded = re.sub(r"(?<=\d),(?=\d)", "", folded)
    folded = re.sub(r"\.(?!\d)", " ", folded)
    return _SPACE.sub(" ", folded).strip()


def contains_phrase(text: str, phrase: str) -> bool:
    """True when ``phrase`` appears on word boundaries inside normalized ``text``."""
    if not phrase:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def tokens(text: str) -> set[str]:
    """Word tokens from normalized text."""
    return set(_TOKEN.findall(normalize(text)))


def parse_decimal(value: str) -> Decimal | None:
    """Parse a warehouse cell. Blank and null become ``None``."""
    text = (value or "").strip()
    if not text or text.lower() == "none":
        return None
    try:
        return Decimal(text)
    except Exception:
        return None


def format_number(value: Decimal) -> str:
    """Format a count or amount. Whole numbers stay whole."""
    quantized = value.quantize(_CENT, rounding=ROUND_HALF_UP)
    if quantized == quantized.to_integral():
        return f"{quantized:,.0f}"
    return f"{quantized:,.2f}"


def format_measure(value: Decimal, scale: str) -> str:
    """Format a measure using its catalog scale.

    A ratio is shown as a percent (0.248 is 24.8%). A percent is already
    on that scale and is not multiplied again.
    """
    if scale == "ratio":
        return _percent(value * Decimal("100"))
    if scale == "percent":
        return _percent(value)
    return format_number(value)


def _percent(value: Decimal) -> str:
    """One decimal when the figure is not already a whole percent.

    76.976 is 77.0%. 100 and 0 stay 100% and 0%.
    """
    precise = value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    quantized = value.quantize(_ONE, rounding=ROUND_HALF_UP)
    if quantized == quantized.to_integral() and precise == precise.to_integral():
        return f"{quantized:,.0f}%"
    return f"{quantized:,.1f}%"


def name_list(names: list[str]) -> str:
    """Join names in a sentence."""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def markdown_table(headers: list[str], rows: list[tuple[str, ...]]) -> str:
    """One GitHub pipe table."""
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, rule, *body])
