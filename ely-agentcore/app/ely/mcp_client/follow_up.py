"""The search text for a question asked partway through a conversation.

Searching with the previous question glued on breaks a change of topic: after "What is Keyman insurance?",
"How many privilege leaves can I carry forward?" found only insurance passages and was answered "not
available". A fast model instead rewrites the question into a standalone search: unchanged for a new topic,
with the missing subject filled in from the previous question for a real follow-up ("and for agency?").
"""
import logging
import os

import boto3

logger = logging.getLogger(__name__)

REWRITE_MODEL_ID = os.getenv("COMPARE_MODEL_ID", "in.anthropic.claude-haiku-4-5-20251001-v1:0")

REWRITE_PROMPT = """You turn the latest question in a chat with an insurance company's assistant into a standalone
search query for its document search.

Previous question: {previous}
Latest question: {current}

If the latest question is complete on its own, or is about a different subject than the previous question,
reply with the latest question exactly as written.
Only if it depends on the previous question (it refers back to it, e.g. "and for agency?", "what about its
entry age?", "is that the same for NRIs?"), rewrite it to include the subject it is missing.
Keep the user's language (English, Hindi or Hinglish). Reply with only the query, nothing else."""

_bedrock = None


def standalone_query(previous: str, current: str) -> str:
    """The question to search for; the current question as written if the model can't be reached."""
    global _bedrock
    if not previous:
        return current
    try:
        if _bedrock is None:
            _bedrock = boto3.client("bedrock-runtime")
        response = _bedrock.converse(
            modelId=REWRITE_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": REWRITE_PROMPT.format(previous=previous, current=current)}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 200},
        )
        query = " ".join(b["text"] for b in response["output"]["message"]["content"] if "text" in b).strip()
    except Exception:
        logger.exception("Could not rewrite the follow-up question; searching it as written")
        return current
    return query.strip('"') or current
