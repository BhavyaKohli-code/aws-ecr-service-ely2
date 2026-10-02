"""The search texts for a question: the question as asked, plus English rewrites worded like the documents.

Searching with the previous question glued on breaks a change of topic: after "What is Keyman insurance?",
"How many privilege leaves can I carry forward?" found only insurance passages and was answered "not
available". A fast model instead rewrites the question into a standalone search: unchanged in subject for a
new topic, with the missing subject filled in from the previous question for a real follow-up ("and for
agency?").

Every question is rewritten, not only follow-ups. The documents are in English and the search matches
meaning only loosely, so a Hinglish question with one key term ("Pep client hai, process kaise hoga?")
found sales-process training instead of the underwriting guideline that answers it. The rewrite translates
it, spells out abbreviations (PEP -> Politically Exposed Person), and adds two phrases worded the way the
documents state answers ("PEP client underwriting: process and requirements"), which is what finds them.
"""
import logging
import os
import re

import boto3

logger = logging.getLogger(__name__)

REWRITE_MODEL_ID = os.getenv("COMPARE_MODEL_ID", "in.anthropic.claude-haiku-4-5-20251001-v1:0")

REWRITE_PROMPT = """You rewrite the latest question in a chat with an insurance company's assistant as one standalone
English question for its document search. The documents are written in English.

Previous question: {previous}
Latest question: {current}

Step 1. Decide whether the latest question is NEW or a FOLLOW_UP.
- NEW: it names its own subject (a product, process, policy or topic), even if the previous question was
  about something related. Use nothing from the previous question.
- FOLLOW_UP: on its own it is missing its subject, so it only makes sense together with the previous
  question: e.g. "and for agency?", "what about its entry age?", "NRI ke liye bhi same hai kya?",
  "aur premium payment term?". Keep the previous question's subject (product name, case) and apply the
  latest question's change to it.

Step 2. Write the question in English: translate Hindi or Hinglish, spell out abbreviations after the short
form (for example "PEP (Politically Exposed Person)"), and use the words the company's documents would use,
such as policy, proposer, owner, life assured, premium, underwriting, eligibility. Keep product names, codes
and numbers as written.

Step 3. Write two more search phrases for the same question, worded the way the company's documents (product
brochures, underwriting guidelines, sales training, FAQs, HR policies) would state the answer, e.g. a section
heading or a table row. Make the first one state it as a rule: "<subject>: <eligibility / underwriting
requirements / documents required / process / limit>", choosing what the question asks for. Use different
key terms in the second.

Reply with exactly four lines: NEW or FOLLOW_UP, the question, then the two phrases. No numbering.
Example: previous "What is the entry age for SWAG?", latest "and for agency?" ->
FOLLOW_UP
What is the entry age for SWAG in the agency channel?
SWAG eligibility: minimum and maximum entry age, agency channel
Smart Wealth Advantage Guarantee plan age at entry for agency"""

# The rewritten question and at most this many phrases are searched, besides the question as asked
MAX_PHRASES = 2

_LABEL = re.compile(r"^(NEW|FOLLOW[_ ]?UP)\W*$", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")

_bedrock = None


def search_queries(previous: str, current: str) -> tuple[str, list[str]]:
    """The standalone English question, and every text to search for: the question as asked first, then the
    rewrites. Only the question as asked if the model can't be reached."""
    global _bedrock
    try:
        if _bedrock is None:
            _bedrock = boto3.client("bedrock-runtime")
        prompt = REWRITE_PROMPT.format(previous=previous or "(none)", current=current)
        response = _bedrock.converse(
            modelId=REWRITE_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 300},
        )
        reply = "\n".join(b["text"] for b in response["output"]["message"]["content"] if "text" in b)
    except Exception:
        logger.exception("Could not rewrite the question; searching it as written")
        return current, [current]
    lines = [_BULLET.sub("", line).strip().strip('"') for line in reply.splitlines()]
    rewrites = [line for line in lines if line and not _LABEL.match(line)][:1 + MAX_PHRASES]
    question = rewrites[0] if rewrites else current
    return question, list(dict.fromkeys([current, *rewrites]))
