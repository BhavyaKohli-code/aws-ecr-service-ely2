"""The Bedrock guardrail every question is checked against before anything is searched.

The guardrail (ely-guardrail in the Bedrock console) blocks questions that aren't about work: pop culture, general
knowledge trivia, sports, politics and religion, recipes and jokes, weather and translation, coding, plus harmful
content and attempts to override the instructions. Its topics, filters and blocked message are set in the console;
here it is only applied.

The question is checked with WORK_PREFIX in front of it. Alone, short work questions read as trivia to the
guardrail ("difference between LOP and LWP?", "latest claim settlement ratio?"); with it they pass, and off-topic
questions are still blocked (sheet test, 2026-10-20: 0 of 200 work questions and 60 of 60 off-topic ones blocked).
A failed check lets the question through: the guardrail must never take the assistant down.
"""
import logging
import os

import boto3

logger = logging.getLogger(__name__)

# No check without a guardrail
GUARDRAIL_ID = os.getenv("GUARDRAIL_ID", "")
GUARDRAIL_VERSION = os.getenv("GUARDRAIL_VERSION", "DRAFT")
WORK_PREFIX = "Work question from an Axis Max Life Insurance employee or advisor: "

_bedrock = None


def blocked_answer(question: str) -> str | None:
    """The guardrail's blocked message if it blocks the question, else None."""
    global _bedrock
    if not GUARDRAIL_ID or not question.strip():
        return None
    try:
        if _bedrock is None:
            _bedrock = boto3.client("bedrock-runtime")
        response = _bedrock.apply_guardrail(
            guardrailIdentifier=GUARDRAIL_ID, guardrailVersion=GUARDRAIL_VERSION, source="INPUT",
            content=[{"text": {"text": WORK_PREFIX + question[:2000]}}])
    except Exception:
        logger.exception("Guardrail check failed; answering the question unchecked")
        return None
    if response.get("action") != "GUARDRAIL_INTERVENED":
        return None
    hits = []
    for assessment in response.get("assessments", []):
        hits += [t["name"] for t in assessment.get("topicPolicy", {}).get("topics", []) if t.get("action") == "BLOCKED"]
        hits += [f["type"] for f in assessment.get("contentPolicy", {}).get("filters", []) if f.get("action") == "BLOCKED"]
        hits += ["profanity" for w in assessment.get("wordPolicy", {}).get("managedWordLists", [])
                 if w.get("action") == "BLOCKED"]
    logger.info("Guardrail blocked the question (%s)", ", ".join(hits) or "unknown policy")
    return "".join(o.get("text", "") for o in response.get("outputs", [])).strip() or None
