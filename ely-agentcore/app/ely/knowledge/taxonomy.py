"""Reads taxonomy.json — the single definition of how ELY's knowledge is organised.

Used by tools/build_metadata.py (tags every S3 document) and by the agent (search filters).
"""
import json
import re
from functools import lru_cache
from pathlib import Path

_TAXONOMY_FILE = Path(__file__).with_name("taxonomy.json")


@lru_cache(maxsize=1)
def taxonomy() -> dict:
    return json.loads(_TAXONOMY_FILE.read_text(encoding="utf-8"))


def label(category: str) -> str:
    return taxonomy()["categories"].get(category, {}).get("label", category)


def domain(category: str) -> str:
    return taxonomy()["categories"].get(category, {}).get("domain", "")


# ---------------------------------------------------------------- documents

def document_attributes(bucket: str, key: str) -> dict:
    """Metadata for one S3 document: category, domain, channel, product, duplicate."""
    t = taxonomy()
    name = key.rsplit("/", 1)[-1]
    attrs = {"category": "", "channel": "all", "product": "none", "duplicate": False}
    for rule in t["folders"]:
        if rule["bucket"] == bucket and key.startswith(rule["prefix"]):
            attrs.update({k: rule[k] for k in ("category", "channel", "duplicate") if k in rule})
            break
    for rule in t["file_overrides"]:
        if (rule.get("bucket", bucket) == bucket and key.startswith(rule.get("prefix", ""))
                and re.search(rule["match"], name, re.I)):
            attrs.update({k: rule[k] for k in ("category", "channel", "duplicate") if k in rule})
    low = name.lower()
    for product_id, product in t["products"].items():
        if any(p in low for p in product["file_patterns"]):
            attrs["product"] = product_id
            break
    attrs["domain"] = domain(attrs["category"])
    return attrs


# ---------------------------------------------------------------- questions

def _alias_regex(alias: str) -> str:
    return rf"(?<![a-z0-9]){re.escape(alias.lower())}(?![a-z0-9])"


@lru_cache(maxsize=1)
def _compiled_rules() -> dict:
    t = taxonomy()
    products = "|".join(_alias_regex(a) for p in t["products"].values() for a in sorted(p["aliases"], key=len, reverse=True))
    competitors = "|".join(_alias_regex(c) for c in t["competitors"])
    rules = {}
    for name in t["intent_rules"]["order"]:
        rule = t["intent_rules"][name]
        pattern = rule["regex"].replace("@products", products).replace("@competitors", competitors)
        rules[name] = {**rule, "compiled": re.compile(pattern, re.I)}
    return rules


def detect_products(question: str) -> list[str]:
    """Product ids named in the question (by any alias), longest alias first."""
    low, found = question.lower(), []
    for product_id, product in taxonomy()["products"].items():
        if any(re.search(_alias_regex(a), low) for a in product["aliases"]):
            found.append(product_id)
    return found


def preferred_category(question: str) -> str | None:
    """The category the old router's tie-breakers point to, or None when no rule applies."""
    rules = _compiled_rules()
    matched = {name for name, rule in rules.items() if rule["compiled"].search(question)}
    for name in taxonomy()["intent_rules"]["order"]:
        rule = rules[name]
        if name in matched and rule.get("unless") not in matched:
            return rule["category"]
    return None


def excluded_data_sources() -> list[str]:
    return taxonomy().get("excluded_data_sources", [])


def knowledge_base_domain(gateway_target: str) -> str:
    return taxonomy()["knowledge_bases"].get(gateway_target, {}).get("domain", "")


def search_settings() -> dict:
    return taxonomy()["search"]
