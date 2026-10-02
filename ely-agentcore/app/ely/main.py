import asyncio
import re
from typing import Any
import boto3
from strands import Agent
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from model.load import BEDROCK_MODEL_ID, accepts_temperature, load_model
from mcp_client.client import GatewayAuth, get_gateway_mcp_client
from mcp_client.contexts import context_options, folder_label, parse_context, passages_by_area
from mcp_client.follow_up import search_queries
from mcp_client.knowledge_search import MAX_SEARCH_RESULTS, SECOND_LOOK_RESULTS, KnowledgeSearch, format_passages
from memory.session import get_memory_session_manager
from model.reply import ASK_AREA_MARKER, MARKERS, NOT_AVAILABLE_MARKER, ReplyFilter
from sources.presign import presign_sources, select_sources

app = BedrockAgentCoreApp()
log = app.logger

DEFAULT_SYSTEM_PROMPT = """
You are ELY, an enterprise employee assistant.

Answer the user's question using ONLY the knowledge base passages provided below. They were retrieved
from the company knowledge bases this user has access to.

Rules:
- Every fact in your answer must come from the passages. Do not add facts, numbers, dates, examples or
  general explanations from your own knowledge, even if you believe they are true.
- If the passages do not contain the answer, start your reply with [NOT_AVAILABLE] and then say clearly
  that this information is not available in the knowledge base. Do not guess, and do not answer
  partially from general knowledge. If the question clearly belongs to a knowledge base the user cannot
  search (see "Knowledge bases this user can search" below; for example a leave or salary question from a
  user who can only search Sales), say that this topic is covered by that other knowledge base, which they
  don't have access to.
- If the passages only cover part of the question, answer that part and say the rest is not available.
- When passages from different documents each cover part of the question, combine them into one complete
  answer (for example a leave policy, its FAQ and the HR handbook together).
- Do not mention passage numbers, tools or knowledge base internals in the answer.
- If the user asks about your previous answer (for example whether it came from HR or Sales), use the
  "Previous answer" note below, which names the documents that answer was built from.
- If the user asks about this conversation itself (what they asked before, how many times, which question
  came first, a summary of the chat), answer from the earlier messages in this conversation, not from the
  passages, and do not mark it [NOT_AVAILABLE]. You can't see when messages were sent, only their order, so
  answer with the order (for example "two questions ago") rather than a time. Then write USED_PASSAGES: none.

Answer naturally and concisely, as an enterprise assistant speaking directly to the employee.

Then, on the very last line, write the numbers of the passages you actually used:
USED_PASSAGES: 2, 5
If you used no passage, write:
USED_PASSAGES: none
"""

NO_ACCESS_ANSWER = "You don't have access to any knowledge base, so I can't answer company questions for you."
NOT_FOUND_ANSWER = "I couldn't find any information about this in the knowledge base."
NO_SOURCE_ANSWER = ("My previous answer wasn't based on any document: the knowledge base didn't have "
                    "information on that question.")
CLARIFY_QUESTION = "I found information on this in more than one area. Which one is your question about?"
# Added to the system prompt when the passages come from more than one area (folder). The model replies
# ASK_AREA_MARKER alone when they answer differently, and the areas are offered as choices.
AREA_RULE = """
The passages come from different areas of the knowledge base:
{areas}
If at least two of these areas each answer the question but say different things for it (different rules,
values, limits, processes, documents or products, so the right answer depends on which area the user means:
for example a different entry age or minimum premium per product or channel, a different process in one
channel than another, or employee incentive plans in one area and advisor sales commission in another),
don't answer: reply with only [ASK_AREA], and the user will be asked to pick an area. If the areas agree, add
detail to one and the same answer, or only one of them answers the question, answer normally.
"""
# The model starts its reply with NOT_AVAILABLE_MARKER when the passages don't answer the question; markers
# are removed before anything reaches the caller (see model/reply.py).
# The model ends its reply with this line naming the passages it used; it is removed before anything
# reaches the caller and decides which documents are returned as sources.
USED_PASSAGES = re.compile(r"\s*USED_PASSAGES:\s*([^\n]*)\s*$", re.I)

# "Which document did you answer that from?" and similar: answered from the sources saved with the
# previous answer, without searching again (a new search would return different documents).
_SOURCE_WORDS = re.compile(r"\b(documents?|docs?|sources?|files?|references?|links?)\b", re.I)
_REFERS_BACK = re.compile(
    r"\b(above|previous|earlier|last answer|that answer|this answer|your answer|did you|you (just )?(used?|answered|got|found|referred|took)"
    r"|says? (that|this)|sources? (of|for) (that|this|it)|(that|this|it) (is |was )?(from|come from|came from)"
    r"|(that|this|the same) (document|doc|file|source|link)s?)\b", re.I)
_WHERE_FROM = re.compile(r"\bwhere (did|do) you (get|find|take|read|see)\b", re.I)
# Answers are kept in session state under this key: {"status": ..., "sources": [...]}
LAST_ANSWER_KEY = "ely_last_answer"
KNOWLEDGE_BASE_LABELS = {"hr-knowledge-retrieval": "HR", "sales-knowledge-retrieval": "Sales"}
# Added to the system prompt in a specialist chat (the caller sent "knowledge_bases"): only those are searched,
# so a question that belongs to another knowledge base is pointed there instead of answered.
SPECIALIST_RULE = """
In this chat the user is talking to the {scope} specialist, which searches only the {scope} knowledge base.
If the question belongs to another knowledge base, start your reply with [NOT_AVAILABLE] and say that the
{scope} specialist only covers {scope} topics; {redirect}
"""


def _no_scope_access_answer(knowledge_bases: list[str]) -> str:
    names = " and ".join(_knowledge_base_label(k) for k in knowledge_bases)
    return f"You don't have access to the {names} knowledge base, so this specialist can't answer for you."


def _specialist_note(knowledge_bases: list[str], accessible: list[str]) -> str:
    """System prompt note for a specialist chat limited to `knowledge_bases`, pointing the user to where a
    question for another knowledge base can be asked (only ones they may use)."""
    scope = " and ".join(_knowledge_base_label(k) for k in knowledge_bases)
    others = [_knowledge_base_label(k) for k in accessible if k not in knowledge_bases]
    if others:
        redirect = (f"suggest asking Ely or the {' or '.join(others)} specialist instead. Don't answer it from "
                    "the passages, and don't mention passages that are not about it.")
    else:
        redirect = ("if it belongs to a knowledge base the user cannot search, say that it is covered by that "
                    "knowledge base, which they don't have access to.")
    return SPECIALIST_RULE.format(scope=scope, redirect=redirect)


def _requested_knowledge_bases(payload) -> list[str] | None:
    """The knowledge bases a specialist chat is limited to ("knowledge_bases" in the payload); None for Ely,
    who searches every knowledge base the user may use."""
    raw = payload.get("knowledge_bases") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return None
    names = [k for k in raw if isinstance(k, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", k)]
    return list(dict.fromkeys(names)) or None

_bedrock = None


def _make_conversation_manager():
    return NullConversationManager()

def agent_factory():
    cache = {}
    def get_or_create_agent(session_id, user_id, authorization):
        """Returns the agent for this session/user, with the caller's current token set for the Gateway."""
        _actor_id = user_id
        key = f"{session_id}/{_actor_id}"
        if key in cache:
            agent, gateway_auth, search = cache[key]
            gateway_auth.authorization = authorization
            return agent, search
        # The token must be set before connecting: the Gateway lists only the tools this
        # user's policies allow, and the search covers exactly that list.
        gateway_auth = GatewayAuth()
        gateway_auth.authorization = authorization
        search = KnowledgeSearch(get_gateway_mcp_client(gateway_auth).start())
        # No tools: retrieval always runs in code before the model is called, so the model
        # cannot skip it and answer from its own knowledge.
        agent = Agent(
            model=load_model(),
            session_manager=get_memory_session_manager(session_id, _actor_id),
            conversation_manager=_make_conversation_manager(),
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            tools=[],
            hooks=[
            ],
        )
        cache[key] = (agent, gateway_auth, search)
        return agent, search
    return get_or_create_agent
get_or_create_agent = agent_factory()


def strip_trailing_tool_use(messages: Any) -> list[dict]:
    """Strip toolUse blocks from the tail until the last message has none."""
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    messages = list(messages)
    while messages:
        last = messages[-1]
        if not isinstance(last, dict):
            raise ValueError("each message must be an object")
        original_content = last.get("content", [])
        if not isinstance(original_content, list) or not all(isinstance(block, dict) for block in original_content):
            raise ValueError("each message content value must be a list of content blocks")

        content = [block for block in original_content if "toolUse" not in block]
        if len(content) == len(original_content):
            break
        if content:
            messages[-1] = {**last, "content": content}
            break
        messages.pop()

    return messages


def _extract_prompt(payload: dict):
    """Accept validated harness messages, tool results, or a plain prompt string."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    if "messages" in payload:
        return strip_trailing_tool_use(payload["messages"])
    if "tool_results" in payload:
        tool_results = payload["tool_results"]
        if not isinstance(tool_results, list) or not all(
            isinstance(tool_result, dict) and isinstance(tool_result.get("toolUseId"), str)
            for tool_result in tool_results
        ):
            raise ValueError("tool_results must contain objects with a toolUseId string")
        return [{"role": "user", "content": [{"toolResult": {
            "toolUseId": tr["toolUseId"],
            "status": tr.get("status", "success"),
            "content": tr.get("content", []),
        }} for tr in tool_results]}]
    # "query" is accepted for callers of the previous ELY runtime
    prompt = payload.get("prompt", payload.get("query", ""))
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    return prompt


def _user_text(message: dict) -> str:
    return " ".join(b["text"] for b in message.get("content", []) if isinstance(b, dict) and b.get("text")).strip()


def _question_and_previous(history: list, prompt) -> tuple[str, str]:
    """The current question and the user's previous one ("" if this is the first)."""
    if isinstance(prompt, str):
        current, earlier = prompt.strip(), history
    else:  # a caller-supplied message list: search for its last user message
        users = [m for m in prompt if m.get("role") == "user" and _user_text(m)]
        current, earlier = (_user_text(users[-1]) if users else ""), history + prompt[:-1]
    previous = next((_user_text(m) for m in reversed(earlier) if m.get("role") == "user" and _user_text(m)), "")
    return current, previous


def _final(answer: str, sources: list, answer_status: str) -> dict:
    """answer_status: 'answered', 'not_available' (knowledge base doesn't cover it) or 'no_access'."""
    return {"type": "final", "answer": answer, "answer_status": answer_status, "sources": sources}


def _clarify(options: list[dict]) -> dict:
    """Final event asking which area the question is about. The caller asks the same question again
    with the chosen option's {knowledge_base, folders} as "context"."""
    return {"type": "final", "answer": CLARIFY_QUESTION, "answer_status": "clarify", "sources": [],
            "options": options}


def _context_label(context: dict, with_knowledge_base: bool) -> str:
    label = " / ".join(folder_label(f) for f in context["folders"])
    return f"{label} ({_knowledge_base_label(context['knowledge_base'])})" if with_knowledge_base else label


def _strip_marker(text: str) -> str:
    for marker in MARKERS:
        text = text.replace(marker, "")
    return text.lstrip(" :-\n").rstrip()


def _areas(results: list[dict]) -> tuple[list[dict], str]:
    """The areas (folders) with good matches among the passages the model reads, as choices, and the system
    prompt note naming each area's passages; none when the matches come from a single area."""
    options = context_options(results)
    numbers = passages_by_area(options, results)
    options = [o for o, n in zip(options, numbers) if n]
    numbers = [n for n in numbers if n]
    if len(options) < 2:
        return [], ""
    several_kbs = len({o["knowledge_base"] for o in options}) > 1
    for option in options:
        option["label"] = _context_label(option, several_kbs)
    areas = "\n".join(f"- {o['label']}: passages {', '.join(map(str, n))}" for o, n in zip(options, numbers))
    return options, AREA_RULE.format(areas=areas) + "\n"


def _replace_reply(agent, text: str) -> None:
    """Replace the reply just saved, in the agent and in memory, with what the caller was given."""
    message = {"role": "assistant", "content": [{"text": text}]}
    agent.messages[-1] = message
    session_manager = getattr(agent, "_session_manager", None)
    if session_manager:
        try:
            session_manager.redact_latest_message(message, agent)
        except Exception:
            log.exception("Could not replace the saved reply in memory")


def _split_used_passages(text: str, count: int) -> tuple[str, list[int] | None]:
    """Remove the trailing USED_PASSAGES line; return the clean text and the 1-based passage numbers used.

    None means the line was missing (the model didn't follow the format); [] means it used no passage.
    """
    match = USED_PASSAGES.search(text)
    if not match:
        return text, None
    used = [int(n) for n in re.findall(r"\d+", match.group(1)) if 1 <= int(n) <= count]
    return text[:match.start()].rstrip(), list(dict.fromkeys(used))


def is_source_question(prompt) -> bool:
    """True for a follow-up asking which document the previous answer came from."""
    if not isinstance(prompt, str):
        return False
    if _WHERE_FROM.search(prompt) and len(prompt.split()) <= 12:
        return True
    if not _SOURCE_WORDS.search(prompt):
        return False
    words = len(prompt.split())
    return words <= 4 or (words <= 15 and bool(_REFERS_BACK.search(prompt)))


def _knowledge_base_label(knowledge_base: str | None) -> str:
    return KNOWLEDGE_BASE_LABELS.get(knowledge_base or "", knowledge_base or "knowledge base")


def _previous_answer_note(last: dict | None) -> str:
    """One line for the system prompt naming the documents the previous answer was built from."""
    if not last:
        return ""
    if not last.get("sources"):
        return "\nPrevious answer: it was not based on any document (the knowledge base did not cover it).\n"
    docs = "; ".join(f"{s['s3_uri'].rsplit('/', 1)[-1]} ({_knowledge_base_label(s.get('knowledge_base'))} knowledge base)"
                     for s in last["sources"])
    return f"\nPrevious answer: it was built from {docs}.\n"


def _source_answer(last: dict) -> tuple[str, list, str]:
    """Answer, fresh sources and status for 'which document did you use?', from the saved previous answer."""
    if not last.get("sources"):
        return NO_SOURCE_ANSWER, [], "not_available"
    sources = presign_sources(last["sources"])
    lines = [f"- {s['title']} ({_knowledge_base_label(s.get('knowledge_base'))} knowledge base)" for s in sources]
    return "My previous answer came from:\n" + "\n".join(lines), sources, "answered"


def _remember_answer(agent, status: str, sources: list) -> None:
    """Save the answer's sources in session state, which AgentCore Memory keeps with the session."""
    keep = ("s3_uri", "score", "snippet", "knowledge_base")
    agent.state.set(LAST_ANSWER_KEY, {"status": status, "sources": [{k: s.get(k) for k in keep} for s in sources]})
    session_manager = getattr(agent, "_session_manager", None)
    if session_manager:  # the state changed after the turn ended, so it needs its own save
        try:
            session_manager.sync_agent(agent)
        except Exception:
            log.exception("Could not save the answer's sources to memory")


def _drop_reasoning(messages: list) -> None:
    """Remove reasoning blocks from earlier replies (reasoning models add them), so only the question and
    answer text are sent back to the model. Memory may still hold them; they are removed again each turn."""
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(isinstance(b, dict) and "reasoningContent" in b for b in content):
            kept = [b for b in content if not (isinstance(b, dict) and "reasoningContent" in b)]
            message["content"] = kept or [{"text": "(no answer)"}]  # Bedrock rejects empty messages


def _second_look(messages: list, system_prompt: str) -> str:
    """The answering model's reply to the same conversation from other passages; "" if it can't be reached.
    Called directly rather than through the agent, so the question isn't added to memory a second time."""
    global _bedrock
    # Only the text of each message: memory adds fields (tracking_id) that Bedrock rejects
    messages = [{"role": m["role"],
                 "content": [{"text": b["text"]} for b in m.get("content", []) if isinstance(b, dict) and b.get("text")]
                 or [{"text": "(no answer)"}]} for m in messages]
    try:
        if _bedrock is None:
            _bedrock = boto3.client("bedrock-runtime")
        settings = {"temperature": 0.0} if accepts_temperature(BEDROCK_MODEL_ID) else {}
        response = _bedrock.converse(modelId=BEDROCK_MODEL_ID, system=[{"text": system_prompt}], messages=messages,
                                     inferenceConfig={"maxTokens": 4096, **settings})
        return "\n".join(b["text"] for b in response["output"]["message"]["content"] if "text" in b)
    except Exception:
        log.exception("Second look failed; keeping the first reply")
        return ""


def _get_authorization(context) -> str | None:
    """The caller's Cognito token, forwarded by Runtime via requestHeaderAllowlist."""
    headers = getattr(context, "request_headers", None) or {}
    for name, value in headers.items():
        if name.lower() == "authorization":
            return value
    return None


@app.entrypoint
async def invoke(payload, context):
    log.info("Invoking Agent.....")


    session_id = getattr(context, 'session_id', 'default-session')
    user_id = getattr(context, 'user_id', 'default-user')
    # Reuse the caller's own token for the Gateway (both use the same Cognito pool)
    authorization = _get_authorization(context)
    if not authorization:
        log.warning("No Authorization header on request; Gateway calls will be unauthenticated")
    agent, search = get_or_create_agent(session_id, user_id, authorization)

    prompt = _extract_prompt(payload)

    # Retrieve first, in code, from every knowledge base this user may use
    if not search.knowledge_bases:
        yield _final(NO_ACCESS_ANSWER, [], "no_access")
        return

    # A specialist chat searches only its own knowledge base(s), and only those the user may use
    requested = _requested_knowledge_bases(payload)
    knowledge_bases = search.knowledge_bases
    if requested is not None:
        knowledge_bases = [k for k in search.knowledge_bases if k in requested]
        if not knowledge_bases:
            yield _final(_no_scope_access_answer(requested), [], "no_access")
            return

    # "Which document was that from?": answer from the saved sources of the previous answer
    last = agent.state.get(LAST_ANSWER_KEY)
    if last and is_source_question(prompt):
        answer, sources, status = _source_answer(last)
        log.info("Source question answered from the previous answer's %d source(s)", len(sources))
        yield {"event": {"contentBlockDelta": {"delta": {"text": answer}}}}
        yield _final(answer, sources, status)
        return

    # The area the user picked after a clarifying question; ignored for a knowledge base they can't use
    context = parse_context(payload.get("context")) if isinstance(payload, dict) else None
    if context and context["knowledge_base"] not in knowledge_bases:
        context = None

    # Search for the question on its own, as asked and rewritten in English the way the documents word it;
    # a follow-up gets the subject it refers to from the previous question
    current, previous = _question_and_previous(agent.messages, prompt)
    _, queries = await asyncio.to_thread(search_queries, previous, current) if current else ("", [])
    results = await search.search(queries, context, knowledge_bases if requested is not None else None) if queries else []
    log.info("Retrieved %d passage(s) from %s for %s (context: %s)", len(results), knowledge_bases, queries, context)
    yield {"type": "retrieval", "knowledge_bases": knowledge_bases, "passages": len(results)}
    if not results:
        _remember_answer(agent, "not_available", [])
        yield _final(NOT_FOUND_ANSWER, [], "not_available")
        return

    # The model reads only the best few; the next ones get a second look if those don't answer the question
    more = results[MAX_SEARCH_RESULTS:MAX_SEARCH_RESULTS + SECOND_LOOK_RESULTS]
    results = results[:MAX_SEARCH_RESULTS]

    # Good matches from more than one folder: the model is told which passages come from which, and replies
    # [ASK_AREA] instead of answering when they answer the question differently. The folders are then offered
    # as choices, and the same question comes back with the chosen one as its context.
    options, area_note = ([], "") if context else _areas(results)

    # Answer from the retrieved passages only. They go in the system prompt for this turn,
    # so conversation memory keeps just the question and the answer.
    context_note = (f"\nThe user has said this question is about: {_context_label(context, True)}.\n"
                    if context else "")
    access_note = ("\nKnowledge bases this user can search: "
                   f"{', '.join(_knowledge_base_label(k) for k in search.knowledge_bases)} "
                   f"(the company has: {', '.join(KNOWLEDGE_BASE_LABELS.values())}).\n")
    specialist_note = _specialist_note(knowledge_bases, search.knowledge_bases) if requested is not None else ""
    instructions = f"{DEFAULT_SYSTEM_PROMPT}{access_note}{specialist_note}{_previous_answer_note(last)}{context_note}\n"
    agent.system_prompt = f"{instructions}{area_note}Knowledge base passages:\n\n{format_passages(results)}"
    reply = ReplyFilter()
    _drop_reasoning(agent.messages)

    async for event in agent.stream_async(
        prompt,
    ):
        if not isinstance(event, dict) or "event" not in event:
            continue
        cbs = event["event"].get("contentBlockStart")
        if cbs is not None and not cbs.get("start"):
            continue
        delta = event["event"].get("contentBlockDelta", {}).get("delta", {})
        if "reasoningContent" in delta:  # the model's private reasoning is never sent to the caller
            continue
        if "text" in delta:
            # Passed on as it arrives, except a reply starting with a marker and the USED_PASSAGES line. Strands
            # saves the piece as changed here, so memory keeps what the caller saw.
            shown, delta["text"] = reply.feed(delta["text"])
            if not shown:
                continue
        yield event

    if reply.marker == ASK_AREA_MARKER and options:
        log.info("Areas %s answer differently: asking which one is meant", [o["label"] for o in options])
        _replace_reply(agent, CLARIFY_QUESTION)
        yield {"event": {"contentBlockDelta": {"delta": {"text": CLARIFY_QUESTION}}}}
        yield _clarify(options)
        return

    if reply.marker:
        # Not answered from these passages, so nothing was shown yet. Search ranks the passage that answers a
        # question anywhere in the first ~25 (often 15th-25th), but the model reads only the best few: rather
        # than send it every passage on every question, it gets one more look, at the next passages, only now.
        answer, used = _split_used_passages(reply.rest(), len(results))
        if more and not used:
            second, second_used = _split_used_passages(await asyncio.to_thread(
                _second_look, agent.messages[:-1], f"{instructions}Knowledge base passages:\n\n{format_passages(more)}"),
                len(more))
            log.info("Second look at %d more passage(s) %s", len(more),
                     "answered the question" if second_used else "didn't answer the question either")
            if second_used:
                results, answer, used = more, second, second_used
        answer = _strip_marker(answer) or NOT_FOUND_ANSWER
        _replace_reply(agent, answer)  # memory saved the reply as the model wrote it
        yield {"event": {"contentBlockDelta": {"delta": {"text": answer}}}}
        # Passages used means the question was (at least partly) answered, even though the model marked it
        not_available = not used
    else:
        answer, used = _split_used_passages(reply.shown + reply.rest(), len(results))
        if used is None and reply.rest():
            # No USED_PASSAGES line: the reply's last line was held back in case it was that line
            yield {"event": {"contentBlockDelta": {"delta": {"text": reply.rest()}}}}
            _replace_reply(agent, answer)
        # The model is told to start with the marker, but may put it later
        not_available = NOT_AVAILABLE_MARKER in answer and not used
        answer = _strip_marker(answer)

    # Final structured event: the answer plus the documents it was built from,
    # each with short-lived pre-signed links. No sources when the passages didn't answer it.
    if not_available:
        log.info("Knowledge base does not cover the question")
        _remember_answer(agent, "not_available", [])
        yield _final(answer, [], "not_available")
        return
    if used == []:
        # Answered without any passage, e.g. "is that from HR or Sales?" answered from the previous-answer
        # note: no sources, and the previous answer's saved sources stay as they are.
        log.info("Answered without using any passage")
        yield _final(answer, [], "answered")
        return
    # Exactly the passages the model used; all of them if it left out the USED_PASSAGES line
    sources = select_sources([results[i - 1] for i in used] if used else results)
    _remember_answer(agent, "answered", sources)
    sources = presign_sources(sources)
    log.info("Answer built from passages %s (%d document(s))", used, len(sources))
    yield _final(answer, sources, "answered")


if __name__ == "__main__":
    app.run()
