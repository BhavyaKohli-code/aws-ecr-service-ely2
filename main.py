
import os
import logging

from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.identity.auth import requires_wat

from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamablehttp_client


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO")
)

logger = logging.getLogger("ely-agent")


# -----------------------------------------------------------------------------
# AgentCore Runtime application
# -----------------------------------------------------------------------------

app = BedrockAgentCoreApp()


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

GATEWAY_URL = os.environ.get(
    "GATEWAY_URL",
    "https://gateway-quick-start-5911f1-51wfgctwi8.gateway.bedrock-agentcore.ap-south-1.amazonaws.com/mcp",
)

BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
)


# -----------------------------------------------------------------------------
# Gateway MCP transport
# -----------------------------------------------------------------------------

def create_gateway_transport(identity_wat: str):
    """
    Create the MCP Streamable HTTP connection to AgentCore Gateway.

    Runtime provides the Workload Access Token (WAT) after validating
    the user's inbound Cognito JWT.

    The WAT is propagated to the downstream Gateway using the
    AgentCore identity propagation header.
    """

    headers = {
        "Authorization": f"Bearer {identity_wat}",
        "X-Amz-Bedrock-AgentCore-Identity-WAT": identity_wat,
        "MCP-Protocol-Version": "2026-07-28",
    }

    logger.info("Connecting to AgentCore Gateway")

    return streamablehttp_client(
        GATEWAY_URL,
        headers=headers,
    )


# -----------------------------------------------------------------------------
# Strands agent
# -----------------------------------------------------------------------------

def create_agent(mcp_client: MCPClient):
    """
    Discover Gateway tools and give them to the Strands agent.

    Claude decides which Gateway tool to use based on the user's
    natural-language request.
    """

    tools = mcp_client.list_tools_sync()

    logger.info(
        "Gateway tools discovered: %s",
        [
            getattr(tool, "tool_name", str(tool))
            for tool in tools
        ],
    )

    model = BedrockModel(
        model_id=BEDROCK_MODEL_ID,
        temperature=0.0,
        streaming=False,
    )

    agent = Agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are ELY, an enterprise employee assistant.\n\n"

            "Your job is to answer the user's question accurately "
            "using the enterprise knowledge available through the "
            "AgentCore Gateway tools.\n\n"

            "The Gateway provides knowledge retrieval tools for "
            "different business domains. Choose the appropriate "
            "tool based on the user's question. Do not ask the "
            "user to specify a tool.\n\n"

            "For HR-related questions, use the HR knowledge "
            "retrieval tool when appropriate.\n\n"

            "For Sales-related questions, use the Sales knowledge "
            "retrieval tool when appropriate.\n\n"

            "Do not invent company policies, procedures, numbers, "
            "dates, or other enterprise facts.\n\n"

            "If the available enterprise knowledge does not contain "
            "enough information to answer the question, clearly say "
            "that the information is not available.\n\n"

            "Answer naturally and concisely, as an enterprise "
            "assistant speaking directly to the employee."
        ),
    )

    return agent


# -----------------------------------------------------------------------------
# AgentCore Runtime invocation
# -----------------------------------------------------------------------------

@app.entrypoint
@requires_wat()
def invoke(payload, identity_wat: str):
    """
    AgentCore Runtime entry point.

    Expected HTTP request body:

        {
            "query": "What is the leave policy?"
        }

    Also accepts:

        {
            "prompt": "What is the leave policy?"
        }

    The caller authenticates to Runtime using a Cognito access token.

    Runtime validates that token and supplies the Workload Access Token
    used for the downstream Gateway call.
    """

    # -------------------------------------------------------------------------
    # Validate payload
    # -------------------------------------------------------------------------

    if not isinstance(payload, dict):
        return {
            "error": "Request body must be a JSON object."
        }

    query = payload.get("query") or payload.get("prompt")

    if not isinstance(query, str) or not query.strip():
        return {
            "error": "Missing 'query' in request body."
        }

    query = query.strip()

    logger.info(
        "Received user query: %s",
        query,
    )

    # -------------------------------------------------------------------------
    # Connect to Gateway
    # -------------------------------------------------------------------------

    mcp_client = MCPClient(
        lambda: create_gateway_transport(identity_wat)
    )

    try:

        # ---------------------------------------------------------------------
        # Discover Gateway tools
        # ---------------------------------------------------------------------

        with mcp_client:

            agent = create_agent(mcp_client)

            # -----------------------------------------------------------------
            # Ask Claude to answer the user's question
            # -----------------------------------------------------------------

            result = agent(query)

            # -----------------------------------------------------------------
            # Return natural-language response
            # -----------------------------------------------------------------

            return {
                "result": result.message
            }

    except Exception as exc:

        logger.exception(
            "ELY agent execution failed"
        )

        return {
            "error": "ELY agent execution failed.",
            "details": str(exc),
        }


# -----------------------------------------------------------------------------
# Local / container entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run()
