from typing import Any
from strands import Agent
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from model.load import load_model
from mcp_client.client import GatewayAuth, get_gateway_mcp_client
from memory.session import get_memory_session_manager

app = BedrockAgentCoreApp()
log = app.logger

DEFAULT_SYSTEM_PROMPT = """
You are ELY, an enterprise employee assistant.

Your job is to answer the user's question accurately using the enterprise knowledge
available through the AgentCore Gateway tools.

The Gateway provides knowledge retrieval tools for different business domains. Choose the
appropriate tool based on the user's question. Do not ask the user to specify a tool.

For HR-related questions, use the HR knowledge retrieval tool when appropriate.

For Sales-related questions, use the Sales knowledge retrieval tool when appropriate.

Do not invent company policies, procedures, numbers, dates, or other enterprise facts.

If the available enterprise knowledge does not contain enough information to answer the
question, clearly say that the information is not available.

Answer naturally and concisely, as an enterprise assistant speaking directly to the employee.
"""


def _make_conversation_manager():
    return NullConversationManager()

def agent_factory():
    cache = {}
    def get_or_create_agent(session_id, user_id, authorization):
        """Returns the agent for this session/user, with the caller's current token set for the Gateway."""
        _actor_id = user_id
        key = f"{session_id}/{_actor_id}"
        if key in cache:
            agent, gateway_auth = cache[key]
            gateway_auth.authorization = authorization
            return agent
        # The token must be set before Agent() is created: creating it connects to
        # the Gateway to list tools.
        gateway_auth = GatewayAuth()
        gateway_auth.authorization = authorization
        agent = Agent(
            model=load_model(),
            session_manager=get_memory_session_manager(session_id, _actor_id),
            conversation_manager=_make_conversation_manager(),
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            tools=[get_gateway_mcp_client(gateway_auth)],
            hooks=[
            ],
        )
        cache[key] = (agent, gateway_auth)
        return agent
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
    agent = get_or_create_agent(session_id, user_id, authorization)

    prompt = _extract_prompt(payload)


    async for event in agent.stream_async(
        prompt,
    ):
        if not isinstance(event, dict) or "event" not in event:
            continue
        cbs = event["event"].get("contentBlockStart")
        if cbs is not None and not cbs.get("start"):
            continue
        yield event


if __name__ == "__main__":
    app.run()
