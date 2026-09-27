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

DEFAULT_SYSTEM_PROMPT = """
You are ELY, an enterprise employee assistant.

Answer the user's question using ONLY the knowledge base passages provided below. They were retrieved
from the company knowledge bases this user has access to.

Rules:
- Every fact in your answer must come from the passages. Do not add facts, numbers, dates, examples or
  general explanations from your own knowledge, even if you believe they are true.
- If the passages do not contain the answer, start your reply with [NOT_AVAILABLE] and then say clearly
  that this information is not available in the knowledge base. Do not guess, and do not answer
  partially from general knowledge.
- If the passages only cover part of the question, answer that part and say the rest is not available.
- Do not mention passage numbers, tools or knowledge base internals.

Answer naturally and concisely, as an enterprise assistant speaking directly to the employee.
"""

NO_ACCESS_ANSWER = "You don't have access to any knowledge base, so I can't answer company questions for you."
NOT_FOUND_ANSWER = "I couldn't find any information about this in the knowledge base."
# The model starts its reply with this when the passages don't answer the question; it is removed
# before anything reaches the caller.
NOT_AVAILABLE_MARKER = "[NOT_AVAILABLE]"


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


def _strip_marker(text: str) -> str:
    return text.replace(NOT_AVAILABLE_MARKER, "").lstrip(" :-\n")


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
    results = await search.search(query) if query else []
    log.info("Retrieved %d passage(s) from %s", len(results), search.knowledge_bases)
    yield {"type": "retrieval", "knowledge_bases": search.knowledge_bases, "passages": len(results)}
    if not results:
        yield _final(NOT_FOUND_ANSWER, [], "not_available")
        return

    # Answer from the retrieved passages only. They go in the system prompt for this turn,
    # so conversation memory keeps just the question and the answer.
    agent.system_prompt = f"{DEFAULT_SYSTEM_PROMPT}\nKnowledge base passages:\n\n{format_passages(results)}"
    answer = ""
    not_available = False

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
        if NOT_AVAILABLE_MARKER in delta.get("text", ""):
            # Stripping here also strips the stored message, so remember it was there
            not_available = True
            delta["text"] = _strip_marker(delta["text"])
        yield event

    # Final structured event: the answer plus the documents it was built from,
    # each with short-lived pre-signed links. No sources when the passages didn't answer it.
    if not_available or NOT_AVAILABLE_MARKER in answer:
        log.info("Knowledge base does not cover the question")
        yield _final(_strip_marker(answer), [], "not_available")
        return
    sources = presign_sources(select_sources(results))
    log.info("Answer built from %d source document(s)", len(sources))
    yield _final(answer, sources, "answered")


if __name__ == "__main__":
    app.run()
