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

The same call decides whether the question may also be searched on the web: only a public fact that changes
over time and that the company, the regulator or the stock exchange publishes (fund returns, NAV, bonus
declared, claim settlement ratio, new fund launches, Nifty). Internal and personal questions (contests, the
user's own sales, weightage grids) and product rules never are: the knowledge base answers those, and a web
page would only give a generic or wrong answer. The knowledge base is still searched for every question,
since the model can't see what it holds (the claim settlement ratio is in the competition benchmarks).
"""
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

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

Step 2. Write the question in English: translate Hindi or Hinglish, spell out the abbreviations listed below
after the short form (for example "PEP (Politically Exposed Person)"), and use the words the company's
documents would use, such as policy, proposer, owner, life assured, premium, underwriting, eligibility. Keep
product names, codes and numbers as written. Spell out only abbreviations from this list; keep any other
abbreviation exactly as written and never guess what it stands for.
{glossary}

Step 3. Write two more search phrases for the same question, worded the way the company's documents (product
brochures, underwriting guidelines, sales training, FAQs, HR policies) would state the answer, e.g. a section
heading or a table row. Make the first one state it as a rule: "<subject>: <eligibility / underwriting
requirements / documents required / process / limit>", choosing what the question asks for. Use different
key terms in the second.

Step 4. Decide whether the question may also be searched on the public web. Today is {today}.
- WEB: it asks for a public fact about Axis Max Life Insurance (formerly Max Life Insurance) or the market that
  changes over time and that the company website, the regulator (IRDAI) or a stock exchange publishes: fund
  performance, returns, NAV, factsheets, fund portfolio or holdings, bonus declared, PAR fund performance,
  claim settlement ratio, AUM, solvency ratio, company results or growth, newly launched funds (NFO),
  market indices (Nifty, Sensex) and their returns.
- KB: anything else, even when it mentions a fund, returns or Nifty. How a product or plan works, its
  benefits, options, illustrated or expected returns ("return % of Smart VIBE for a 39 year old", "in Smart
  RISE how do I get Nifty 50 returns"); eligibility and rules; underwriting; where or how to find something
  ("how can I find fund values"); what a term means ("what is benchmark return"); HR; contests, incentives,
  targets, product weightage or weightage grids, GPA; the user's own data (my sales, my performance, my
  policies); calculations, projections, future returns or advice for a case ("what will I get if I invest
  2 lakh", "which fund gives more returns after 15 years", "ULIP returns in the next 30 years", "make a
  portfolio of 5 lakh"); greetings and chat about the assistant.
- A FOLLOW_UP is WEB or KB as its rewritten question is.
Then write a web search query for a WEB question: short, in English, with the month and year it is about (the
latest one if the question says latest, current or this year). Name Axis Max Life in it for a fact about the
company or its funds, but not for a market index ("Nifty 50 index returns 2010 to October 2026"). For a KB
question write "none".

Reply with exactly six lines: NEW or FOLLOW_UP, the question, the two phrases, then WEB or KB, then the web
search query. No numbering.
Example: previous "What is the entry age for SWAG?", latest "and for agency?" ->
FOLLOW_UP
What is the entry age for SWAG in the agency channel?
SWAG eligibility: minimum and maximum entry age, agency channel
Smart Wealth Advantage Guarantee plan age at entry for agency
KB
none"""

# The company's own abbreviations. Without them the model guessed wrong ("PL requests" became "Personal Loan
# requests" instead of Privilege Leave) and the search found nothing about leave.
GLOSSARY = """- HR: PL = Privilege Leave, CL = Casual Leave, SL = Sick Leave, LOP = Loss of Pay, LWP = Leave Without Pay,
  HRBP = HR Business Partner, LWD = Last Working Day, F&F = Full and Final settlement
- Underwriting and compliance: PEP = Politically Exposed Person, EDD = Enhanced Due Diligence, AFYP = Annualised
  First Year Premium, TSAR = Total Sum At Risk, NRI = Non-Resident Indian, KYC = Know Your Customer,
  AML = Anti-Money Laundering, NTU = Not Taken Up, ULIP = Unit Linked Insurance Plan, Par = Participating
- Products: OSP = Online Savings Plan, OSPP = Online Savings Plan Plus, FTS = Fast Track Super Plan,
  FWAP = Flexi Wealth Advantage Plan, PWP = Platinum Wealth Plan, SPS = Shiksha Plus Super, SRP = Smart RISE,
  STAR ULIP = Smart Term with Additional Returns ULIP, STEP = Smart Total Elite Protection, STPP = Smart Term
  Plan Plus, SWAG = Smart Wealth Advantage Guarantee Plan, SWAG Pension = Smart Wealth Annuity Guaranteed
  Pension Plan, SWAG-Par = Smart Wealth Advantage Growth Par Plan"""

# The rewritten question and at most this many phrases are searched, besides the question as asked
MAX_PHRASES = 2

_LABEL = re.compile(r"^(NEW|FOLLOW[_ ]?UP)\W*$", re.I)
_SOURCE = re.compile(r"^(WEB|KB)\W*$", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
# "Today is ..." in the prompt: India time, so "this year" and "latest" mean the same as for the user
IST = timezone(timedelta(hours=5, minutes=30))

_bedrock = None


_FOLLOW_UP = re.compile(r"^FOLLOW[_ ]?UP\b", re.I)


class SearchPlan(NamedTuple):
    is_follow_up: bool
    queries: list[str]
    # The web search query when the question may also be searched on the web, else None
    web_query: str | None


def search_queries(previous: str, current: str) -> SearchPlan:
    """Whether the question is a follow-up (continues the previous question's subject), every text to search
    the knowledge bases for (the question as asked first, then the rewrites), and the web search query if it
    may be searched on the web. Only the question as asked, not a follow-up and no web search, if the model
    can't be reached.

    The follow-up flag lets the caller skip the "which area?" clarifying step: a follow-up's subject is
    already fixed by the conversation, so asking which area it is about just repeats what we already know.
    """
    global _bedrock
    try:
        if _bedrock is None:
            _bedrock = boto3.client("bedrock-runtime")
        prompt = REWRITE_PROMPT.format(previous=previous or "(none)", current=current, glossary=GLOSSARY,
                                       today=datetime.now(IST).strftime("%d %B %Y"))
        response = _bedrock.converse(
            modelId=REWRITE_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 400},
        )
        reply = "\n".join(b["text"] for b in response["output"]["message"]["content"] if "text" in b)
    except Exception:
        logger.exception("Could not rewrite the question; searching it as written")
        return SearchPlan(False, [current], None)
    lines = [line for line in (_BULLET.sub("", line).strip().strip('"') for line in reply.splitlines()) if line]
    is_follow_up = any(_FOLLOW_UP.match(line) for line in lines)
    # The web query is the line after WEB; a reply without the WEB/KB line is not searched on the web
    source = next((i for i, line in enumerate(lines) if _SOURCE.match(line)), None)
    web_query = None
    if source is not None and lines[source].upper().startswith("WEB") and source + 1 < len(lines):
        web_query = lines[source + 1][:200] if lines[source + 1].lower() != "none" else None
    rewrites = [line for line in lines[:source] if not _LABEL.match(line)][:1 + MAX_PHRASES]
    return SearchPlan(is_follow_up, list(dict.fromkeys([current, *rewrites])), web_query)
