"""Conversational BI: the agency sales figures in Redshift, behind the Gateway's bi-warehouse tool.

The bi-warehouse___query_warehouse tool (Lambda ely-bi-warehouse) reads the figures catalog, writes one read-only
query and returns either a finished markdown answer, a clarifying question (several offices or people match), or a
small row table. A finished answer or a question is shown as it is; the rows, or nothing, go to the model with the
KPI catalogue, which also answers definitions ("what is WFYP?"). The Gateway's policy engine lists the tool only for
full-time users outside the sales channels, so a chat without it has no access.
"""

import json
import logging
import os
import re
import uuid
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)

BI_KNOWLEDGE_BASE = "bi-warehouse"
# Earlier questions the standalone rewrite sees
PRIOR_QUESTIONS = 3
REWRITE_MODEL_ID = os.getenv("COMPARE_MODEL_ID", "in.anthropic.claude-haiku-4-5-20251001-v1:0")

# The figures engine reads earlier questions together with the new one, so given them it answered "What is MDRT?"
# and "same for 25th month" with the previous question's figures. It gets one standalone question instead.
STANDALONE_PROMPT = """You prepare the latest question in a chat about agency sales figures for the figures lookup.

Previous questions, oldest first:
{previous}
Latest question: {current}

Reply with exactly one line:
- DEFINITION if the latest question only asks what a measure, term or rule means or how it is calculated, and
  wants no figure.
- CHART if it only asks to show, chart, plot, graph or visualise the figures already given, or to rework that
  answer ("chart this", "as a pie chart", "shorter"), and names no new figure.
- Otherwise the latest question as one standalone question. If it names its own measure or subject, repeat it
  unchanged. If it is a fragment ("and for direct agency?", "same for 25th month", "only the top 3", "and YTD?"),
  rewrite the most recent previous question with only that change applied. Keep the person's own words and terms
  (WFYP, NOP, MDRT, leaders, month to date); don't explain, expand or translate them."""

# In Ely's own chat, for a user who may use the figures: does this question want them, or the documents?
ROUTE_PROMPT = """An insurance company's assistant can answer from documents (HR policies, products, sales process, KPI
definitions) or from the live agency sales figures (the actual numbers for agents, leaders and offices: paid WFYP,
NOP, FYC, case size, targets and achievement, persistency %, active agents, MDRT and promotion status, rankings,
counts, lists, comparisons).

Previous questions, oldest first:
{previous}
Latest question: {current}

Reply FIGURES if the latest question asks for those actual numbers, people or rankings, or is a follow-up or "chart
this" on an answer that gave them. Reply DOCS for anything else, including what a measure means or how it is
calculated. One word."""

_bedrock = None

KPI_CATALOGUE = (Path(__file__).parent / "agency_kpi_catalogue.txt").read_text(encoding="utf-8")
# The POC harness's echarts-reporting skill, without its front matter; the widget draws its echarts-json blocks
CHART_RULES = (Path(__file__).parent / "echarts_reporting.md").read_text(encoding="utf-8").split("---", 2)[-1].strip()
# A question that asks for a chart goes to the model even when the figures are already a finished answer
_CHART_WORDS = re.compile(r"\b(chart|graph|plot|visuali[sz]e|pie|bar|line|trend|diagram)s?\b", re.I)

NO_BI_ACCESS_ANSWER = "You don't have access to Conversational BI, so I can't look up the sales figures for you."
BI_ERROR_ANSWER = "I couldn't look up the figures just now. Please try again in a moment."

BI_SYSTEM_PROMPT = """
You are the Conversational BI assistant of Axis Max Life Insurance: a senior data analyst for the agency sales
figures (production, targets and achievement, persistency, active agents, MDRT, promotions). People ask in business
language. They do not know how anything is stored.

Answer from the "Figures" section and the "Definitions" section below only:
- A number, ranking, list or comparison comes from the figures. Never answer one from memory and never invent a
  number, person or row. If the figures are empty or don't answer it, say you could not find that in plain words.
- A definition, formula or rule comes from the definitions. Don't expand an acronym they don't define; say it isn't
  defined, without saying where you looked.
- Greetings and questions about this conversation are answered without either.
- "Chart this" or "shorter" reworks the figures already given in this conversation.

Reply in Markdown: a ## headline with the answer, one sentence, then one pipe table for a ranking or a comparison.
Keep the row order of the figures. Give amounts in rupees. Show a ratio as a percentage (0.248 is 24.8%); an
achievement % is already a percentage. State any assumption (period, office) in one sentence.

When they ask for a chart, follow the chart rules below and reply with a ## title and one echarts-json block. Chart
the same figures, in the same order and scale as the answer (percentages as percentages).

Never mention how the answer was produced. Forbidden in the reply: database, warehouse, schema, table, column, SQL,
query, Redshift, knowledge base, catalogue, ECharts, tool names and internal codes such as kpi186 or master_flag;
use the business names (13th-month persistency, leaders, month to date).
"""


def warehouse_tool(tool_names: list[str]) -> str | None:
    """The Gateway's BI tool among those this user may call; None without access."""
    return next((n for n in tool_names if n.split("___")[0] == BI_KNOWLEDGE_BASE), None)


def _payload(outcome: dict) -> dict:
    """The Lambda's reply ({answer, clarification, guidance, rowCount, ...}) from an MCP tool result."""
    structured = outcome.get("structuredContent")
    if isinstance(structured, dict) and ("answer" in structured or "guidance" in structured):
        return structured
    for block in outcome.get("content") or []:
        text = block.get("text") if isinstance(block, dict) else None
        if not text:
            continue
        try:
            payload = json.loads(text)
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def wants_figures(previous: list[str], current: str) -> bool:
    """True when a question in Ely's own chat should be answered from the figures; False (the documents) if the model
    can't be reached."""
    global _bedrock
    try:
        if _bedrock is None:
            _bedrock = boto3.client("bedrock-runtime")
        prompt = ROUTE_PROMPT.format(previous="\n".join(previous[-PRIOR_QUESTIONS:]) or "(none)", current=current)
        response = _bedrock.converse(modelId=REWRITE_MODEL_ID,
                                     messages=[{"role": "user", "content": [{"text": prompt}]}],
                                     inferenceConfig={"temperature": 0.0, "maxTokens": 5})
        reply = "".join(b["text"] for b in response["output"]["message"]["content"] if "text" in b)
    except Exception:
        logger.exception("Could not decide between the figures and the documents; using the documents")
        return False
    figures = reply.strip().upper().startswith("FIGURES")
    logger.info("Ely question %r goes to the %s", current, "figures" if figures else "documents")
    return figures


def wants_chart(question: str) -> bool:
    return bool(_CHART_WORDS.search(question))


def plan_question(previous: list[str], current: str) -> tuple[str, str | None]:
    """("figures", the question standing on its own), ("definition", None) answered from the KPI catalogue, or
    ("chart", None) to rework the figures already given. The question as asked if the model can't be reached, or
    for the first question."""
    global _bedrock
    if not previous:
        return "figures", current
    try:
        if _bedrock is None:
            _bedrock = boto3.client("bedrock-runtime")
        prompt = STANDALONE_PROMPT.format(previous="\n".join(previous[-PRIOR_QUESTIONS:]), current=current)
        response = _bedrock.converse(modelId=REWRITE_MODEL_ID,
                                     messages=[{"role": "user", "content": [{"text": prompt}]}],
                                     inferenceConfig={"temperature": 0.0, "maxTokens": 200})
        reply = "".join(b["text"] for b in response["output"]["message"]["content"] if "text" in b)
    except Exception:
        logger.exception("Could not rewrite the BI question; asking it as written")
        return "figures", current
    line = next((l.strip().strip('"') for l in reply.splitlines() if l.strip()), "")
    for kind in ("definition", "chart"):
        if line.upper().startswith(kind.upper()):
            logger.info("BI question %r: %s", current, kind)
            return kind, None
    logger.info("BI question %r asked as %r", current, line or current)
    return "figures", line or current


async def query_warehouse(mcp_client, tool_name: str, question: str) -> dict:
    """Ask the figures; raises when the tool call fails."""
    outcome = await mcp_client.call_tool_async(f"bi-{uuid.uuid4().hex[:12]}", tool_name, {"question": question})
    if outcome.get("status") == "error":
        raise RuntimeError(f"Warehouse tool failed: {outcome.get('content')}")
    payload = _payload(outcome)
    logger.info("Figures for %r: rows=%s, clarification=%s, resolved=%s", question, payload.get("rowCount"),
                bool(payload.get("clarification")), is_resolved(payload))
    return payload


def is_resolved(payload: dict) -> bool:
    """True when the figures engine already wrote the answer (shown as it is, without the model)."""
    return bool(payload.get("answer")) and "already resolved" in (payload.get("guidance") or "")


def figures_prompt(payload: dict | None) -> str:
    """BI system prompt for this turn, with the figures the tool returned and the KPI catalogue. None: no new figures
    were asked (a definition, or a chart of the answer already in the conversation)."""
    if payload is None:
        figures = "none for this turn: use the figures already given in this conversation, if any."
    else:
        guidance = (payload.get("guidance") or "").strip()
        answer = (payload.get("answer") or "").strip() or "No matching figures."
        figures = f"\n{guidance}\n\n{answer}" if guidance else f"\n{answer}"
    return (f"{BI_SYSTEM_PROMPT}\nFigures for this question: {figures}\n\nChart rules:\n\n{CHART_RULES}\n\n"
            f"Definitions:\n\n{KPI_CATALOGUE}")
