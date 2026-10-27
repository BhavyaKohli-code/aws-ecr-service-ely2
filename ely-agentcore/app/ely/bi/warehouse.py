"""Conversational BI: the agency sales figures in Redshift, behind the Gateway's bi-warehouse tool.

The bi-warehouse___query_warehouse tool (Lambda ely-bi-warehouse) reads the figures catalog, writes one read-only
query and returns either a finished markdown answer, a clarifying question (several offices or people match), or a
small row table. A finished answer or a question is shown as it is; the rows, or nothing, go to the model with the
KPI catalogue, which also answers definitions ("what is WFYP?"). The Gateway's policy engine lists the tool only for
full-time users outside the sales channels, so a chat without it has no access.
"""

import json
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

BI_KNOWLEDGE_BASE = "bi-warehouse"
# Earlier questions sent with each one, so a fragment ("for direct agency", "and YTD?") keeps its subject
PRIOR_QUESTIONS = 3

KPI_CATALOGUE = (Path(__file__).parent / "agency_kpi_catalogue.txt").read_text(encoding="utf-8")

NO_BI_ACCESS_ANSWER = "You don't have access to Conversational BI, so I can't look up the sales figures for you."
BI_ERROR_ANSWER = "I couldn't look up the figures just now. Please try again in a moment."

BI_SYSTEM_PROMPT = """
You are the Conversational BI assistant of Axis Max Life Insurance: a senior data analyst for the agency sales
figures (production, targets and achievement, persistency, active agents, MDRT, promotions). People ask in business
language. They do not know how anything is stored.

Answer from the "Figures" section and the "KPI catalogue" section below only:
- A number, ranking, list or comparison comes from the figures. Never answer one from memory and never invent a
  number, person or row. If the figures are empty or don't answer it, say you could not find that in plain words.
- A definition, formula or rule comes from the KPI catalogue. Don't expand an acronym the catalogue doesn't define.
- Greetings and questions about this conversation are answered without either.
- "Chart this" or "shorter" reworks the figures already given in this conversation.

Reply in Markdown: a ## headline with the answer, one sentence, then one pipe table for a ranking or a comparison.
Keep the row order of the figures. Give amounts in rupees. Show a ratio as a percentage (0.248 is 24.8%); an
achievement % is already a percentage. State any assumption (period, office) in one sentence.

Never mention how the answer was produced. Forbidden in the reply: database, warehouse, schema, table, column, SQL,
query, Redshift, knowledge base, catalogue, tool names and internal codes such as kpi186 or master_flag; use the
business names (13th-month persistency, leaders, month to date).
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


async def query_warehouse(mcp_client, tool_name: str, question: str, prior: list[str]) -> dict:
    """Ask the figures; raises when the tool call fails."""
    outcome = await mcp_client.call_tool_async(f"bi-{uuid.uuid4().hex[:12]}", tool_name,
                                               {"question": question, "prior": prior[-PRIOR_QUESTIONS:]})
    if outcome.get("status") == "error":
        raise RuntimeError(f"Warehouse tool failed: {outcome.get('content')}")
    payload = _payload(outcome)
    logger.info("Figures for %r: rows=%s, clarification=%s, resolved=%s", question, payload.get("rowCount"),
                bool(payload.get("clarification")), is_resolved(payload))
    return payload


def is_resolved(payload: dict) -> bool:
    """True when the figures engine already wrote the answer (shown as it is, without the model)."""
    return bool(payload.get("answer")) and "already resolved" in (payload.get("guidance") or "")


def figures_prompt(payload: dict) -> str:
    """BI system prompt for this turn, with the figures the tool returned (or none) and the KPI catalogue."""
    figures = (payload.get("answer") or "").strip() or "No matching figures."
    guidance = (payload.get("guidance") or "").strip()
    note = f"\n{guidance}\n" if guidance else ""
    return (f"{BI_SYSTEM_PROMPT}\nFigures for this question:{note}\n{figures}\n\n"
            f"KPI catalogue:\n\n{KPI_CATALOGUE}")
