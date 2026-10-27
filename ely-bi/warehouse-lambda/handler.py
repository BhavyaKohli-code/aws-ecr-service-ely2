"""Gateway tool: answer a business question from the figures in Redshift.

Business meaning (who a row is, how to show a percent, how to order a list)
lives in the harness skill and system prompt. This function only reads the
current tables, writes one read-only query, and returns the rows.
"""

from __future__ import annotations

import os
import re
import time

import boto3

_FORBIDDEN = ("insert ", "update ", "delete ", "drop ", "alter ", "create ", "grant ", "truncate ")


def _clients():
    region = os.environ.get("BEDROCK_REGION", "ap-southeast-2")
    return (
        boto3.client("redshift-data", region_name=region),
        boto3.client("bedrock-runtime", region_name=region),
    )


def _run_sql(data_client, sql: str) -> tuple[list[str], list[list[str]]]:
    workgroup = os.environ.get("REDSHIFT_WORKGROUP", "dono")
    database = os.environ.get("REDSHIFT_DATABASE", "dev")
    started = data_client.execute_statement(WorkgroupName=workgroup, Database=database, Sql=sql)
    statement_id = started["Id"]
    # Redshift Serverless can stay PICKED while a paused workgroup resumes.
    deadline = time.time() + 100
    status = "SUBMITTED"
    error = ""
    while time.time() < deadline:
        described = data_client.describe_statement(Id=statement_id)
        status = described.get("Status", "")
        if status in ("FINISHED", "FAILED", "ABORTED"):
            error = described.get("Error") or ""
            break
        time.sleep(0.4)
    if status != "FINISHED":
        raise RuntimeError(error or f"Query did not finish ({status})")
    result = data_client.get_statement_result(Id=statement_id)
    columns = [col["name"] for col in result.get("ColumnMetadata") or []]
    rows: list[list[str]] = []
    for record in result.get("Records") or []:
        values = []
        for cell in record:
            values.append("" if cell.get("isNull") else str(next(iter(cell.values()))))
        rows.append(values)
    return columns, rows


def _column_catalogue(question: str) -> str:
    """Columns for the tables this question needs, not the whole warehouse."""
    from app.semantic.catalog import catalogue_hint

    return catalogue_hint(question)


def _format_table(columns: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "No matching figures."
    header = " | ".join(columns)
    lines = [header]
    for row in rows[:50]:
        lines.append(" | ".join(row))
    return "\n".join(lines)


def _safe_select(sql: str) -> str:
    text = sql.strip()
    text = re.sub(r"^```(?:sql)?", "", text, flags=re.I).strip()
    text = text.replace("```", "").strip()
    text = text.split(";")[0].strip()
    lowered = f" {text.lower()} "
    if not (lowered.strip().startswith("select") or lowered.strip().startswith("with")):
        raise ValueError("Only a read query is allowed")
    if any(token in lowered for token in _FORBIDDEN):
        raise ValueError("Only a read query is allowed")
    text = re.sub(r"\blimit\s+\d+\s*", "", text, flags=re.I).strip()
    return text + " LIMIT 25"


def _generate_sql(runtime, question: str, catalogue: str, scope: str) -> str:
    model_id = os.environ.get("BEDROCK_CHAT_MODEL", "amazon.nova-pro-v1:0")
    prompt = (
        "Write one Amazon Redshift SELECT for the question. "
        "Use only tables and columns from the catalogue. "
        "Do not invent columns. Output only the SQL, no explanation. "
        "The scope is already resolved. Do not treat a business phrase such as "
        "'direct agency' as a person's name or an office name.\n\n"
        f"Scope:\n{scope or 'Use the latest month and month to date.'}\n\n"
        f"Catalogue:\n{catalogue}\n\nQuestion: {question}"
    )
    response = runtime.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 600, "temperature": 0},
    )
    parts = response.get("output", {}).get("message", {}).get("content") or []
    return "".join(part.get("text", "") for part in parts)


def lambda_handler(event, context):  # noqa: ARG001
    question = (event or {}).get("question", "").strip()
    if not question:
        return {"error": "A question is required", "answer": ""}
    prior = (event or {}).get("prior") or []
    if isinstance(prior, str):
        prior = [prior]

    from app.semantic.engine import answer_question

    data_client, runtime = _clients()

    def run_sql(sql: str) -> tuple[list[str], list[list[str]]]:
        return _run_sql(data_client, sql)

    try:
        resolved = answer_question(question, list(prior), run_sql)
    except Exception:
        resolved = None

    if resolved is not None and resolved.status == "answered":
        payload = {
            "answer": resolved.markdown,
            "rowCount": resolved.row_count,
            "error": "",
            "guidance": (
                "These figures are already resolved. Reply with them. "
                "Do not ask a follow-up. Do not say the figures are missing. "
                "Ignore the sql field. Do not quote it."
            ),
        }
        sql = (resolved.sql or "").strip()
        if sql:
            payload["sql"] = sql
        return payload
    if resolved is not None and resolved.status == "clarification":
        return {
            "answer": resolved.markdown,
            "rowCount": 0,
            "error": "",
            "clarification": resolved.markdown,
            "guidance": "Ask the user only this question. Do not say the figures are missing.",
        }

    scope = resolved.brief if resolved is not None else ""
    try:
        catalogue = _column_catalogue(question)
        sql = _safe_select(_generate_sql(runtime, question, catalogue, scope))
        try:
            columns, rows = _run_sql(data_client, sql)
        except Exception:
            columns, rows = [], []
        if not rows:
            sql = _safe_select(
                _generate_sql(
                    runtime,
                    question + " Use only catalogue columns. Return the rows that match the question.",
                    catalogue,
                    scope,
                )
            )
            columns, rows = _run_sql(data_client, sql)
        rows = [row for row in rows if any(cell.strip() for cell in row)]
    except Exception:
        return {
            "answer": "",
            "error": "",
            "clarification": "",
            "guidance": "No matching figures for that question.",
        }

    return {
        "answer": _format_table(columns, rows),
        "sql": sql,
        "rowCount": len(rows),
        "error": "",
        "guidance": (
            "These are the only figures you may cite. "
            "Follow the agency-kpi-analyst skill for names, percentages, order, and wording. "
            "Do not mention tables, columns, SQL, or systems. "
            "Ignore the sql field. Do not quote it. "
            "State any assumption from the scope in one sentence. "
            "If there are no rows, say you could not find that and do not invent numbers. "
            "Do not ask which office when the scope already names one."
        ),
    }
