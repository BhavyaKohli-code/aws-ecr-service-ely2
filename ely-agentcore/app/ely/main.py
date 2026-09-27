import re
from pathlib import Path
from typing import Any
from strands import Agent
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from model.load import load_model
from mcp_client.client import GatewayAuth, get_gateway_mcp_client
from mcp_client.knowledge_search import KnowledgeSearch, format_passages
from memory.session import get_memory_session_manager
from sources.presign import presign_sources, select_sources

app = BedrockAgentCoreApp()
log = app.logger

DEFAULT_SYSTEM_PROMPT = (Path(__file__).parent / "knowledge" / "system_prompt.md").read_text(encoding="utf-8")

NO_ACCESS_ANSWER = "You don't have access to any knowledge base, so I can't answer company questions for you."
NOT_FOUND_ANSWER = "I couldn't find any information about this in the knowledge base."
# The model ends its reply with this line naming the passages it used; it is removed before anything
# reaches the caller and decides which documents are returned as sources.
USED_PASSAGES = re.compile(r"\s*USED_PASSAGES:\s*([^\n]*)\s*$", re.I)


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


def _retrieval_query(history: list, prompt) -> str:
    """The text to search for: the current question, prefixed by the previous one for follow-ups."""
    if isinstance(prompt, str):
        current, earlier = prompt.strip(), history
    else:  # a caller-supplied message list: search for its last user message
        users = [m for m in prompt if m.get("role") == "user" and _user_text(m)]
        current, earlier = (_user_text(users[-1]) if users else ""), history + prompt[:-1]
    previous = next((_user_text(m) for m in reversed(earlier) if m.get("role") == "user" and _user_text(m)), "")
    return f"{previous}\n{current}" if previous else current


def _final(answer: str, sources: list, answer_status: str) -> dict:
    """answer_status: 'answered', 'not_available' (knowledge base doesn't cover it) or 'no_access'."""
    return {"type": "final", "answer": answer, "answer_status": answer_status, "sources": sources}


def _split_used_passages(text: str, count: int) -> tuple[str, list[int]]:
    """Remove the trailing USED_PASSAGES line; return the clean text and the 1-based passage numbers used."""
    match = USED_PASSAGES.search(text)
    if not match:
        return text, []
    used = [int(n) for n in re.findall(r"\d+", match.group(1)) if 1 <= int(n) <= count]
    return text[:match.start()].rstrip(), list(dict.fromkeys(used))


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
    query = _retrieval_query(agent.messages, prompt)
    results, category = await search.search(query) if query else ([], None)
    log.info("Retrieved %d passage(s) from %s (category %s)", len(results), search.knowledge_bases, category)
    yield {"type": "retrieval", "knowledge_bases": search.knowledge_bases, "category": category, "passages": len(results)}
    if not results:
        yield _final(NOT_FOUND_ANSWER, [], "not_available")
        return

    # Answer from the retrieved passages only. They go in the system prompt for this turn,
    # so conversation memory keeps just the question and the answer.
    agent.system_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\nKnowledge base passages:\n\n{format_passages(results)}"
    answer, used = "", []

    # The model is non-streaming (see model/load.py), so its whole reply, including the trailing
    # USED_PASSAGES line, arrives in one text delta. Editing the delta also edits the stored message.
    async for event in agent.stream_async(
        prompt,
    ):
        if isinstance(event, dict) and "result" in event:
            answer = str(event["result"]).strip()
        if not isinstance(event, dict) or "event" not in event:
            continue
        cbs = event["event"].get("contentBlockStart")
        if cbs is not None and not cbs.get("start"):
            continue
        delta = event["event"].get("contentBlockDelta", {}).get("delta", {})
        if "USED_PASSAGES" in delta.get("text", "").upper():
            delta["text"], used = _split_used_passages(delta["text"], len(results))
        yield event

    # Final structured event: the answer plus exactly the documents it used, each with short-lived
    # pre-signed links. No passages used means the knowledge base doesn't cover the question.
    answer, _ = _split_used_passages(answer, len(results))
    if not used:
        log.info("Knowledge base does not cover the question")
        yield _final(answer, [], "not_available")
        return
    sources = presign_sources(select_sources([results[i - 1] for i in used]))
    log.info("Answer built from passages %s (%d document(s))", used, len(sources))
    yield _final(answer, sources, "answered")


if __name__ == "__main__":
    app.run()
