
import sys

# Force unbuffered stdout/stderr so these startup markers (and any
# exception traceback) actually reach CloudWatch instead of sitting in a
# buffer that never gets flushed if the process is killed for taking too
# long to initialize.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import os
import logging
import time

START_TIME = time.perf_counter()

print("=== ELY STARTING ===", flush=True)

from bedrock_agentcore import BedrockAgentCoreApp

print(
    f"bedrock_agentcore imported: {time.perf_counter() - START_TIME:.2f}s",
    flush=True,
)

from bedrock_agentcore.identity.auth import requires_wat

print(
    f"requires_wat imported: {time.perf_counter() - START_TIME:.2f}s",
    flush=True,
)

from strands import Agent

print(
    f"strands Agent imported: {time.perf_counter() - START_TIME:.2f}s",
    flush=True,
)

from strands.models import BedrockModel

print(
    f"BedrockModel imported: {time.perf_counter() - START_TIME:.2f}s",
    flush=True,
)

from strands.tools.mcp.mcp_client import MCPClient

print(
    f"MCPClient imported: {time.perf_counter() - START_TIME:.2f}s",
    flush=True,
)

from mcp.client.streamable_http import streamablehttp_client

print(
    f"MCP transport imported: {time.perf_counter() - START_TIME:.2f}s",
    flush=True,
)

print(
    f"=== ALL IMPORTS COMPLETE: {time.perf_counter() - START_TIME:.2f}s ===",
    flush=True,
)


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
    "mistral.ministral-3-8b-instruct",
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

    start = time.perf_counter()

    logger.info("Starting Gateway tool discovery")

    tools = mcp_client.list_tools_sync()

    logger.info(
        "Gateway tool discovery completed in %.2fs",
        time.perf_counter() - start,
    )

    logger.info(
        "Gateway tools discovered: %s",
        [
            getattr(tool, "tool_name", str(tool))
            for tool in tools
        ],
    )

    logger.info("Creating BedrockModel")

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

            start = time.perf_counter()

            logger.info("Calling ELY agent")

            result = agent(query)

            logger.info(
                "ELY agent completed in %.2fs",
                time.perf_counter() - start,
            )

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
    # Bind on all interfaces: AgentCore Runtime reaches the agent on port 8080.
    # (app.run() only auto-binds 0.0.0.0 inside Docker, which S3 code
    # deployments are not.)
    try:
        print(
            f"=== STARTING APP: {time.perf_counter() - START_TIME:.2f}s ===",
            flush=True,
        )
        app.run(host="0.0.0.0", port=8080)
    except Exception:
        import traceback
        print("=== APP FAILED TO START ===", flush=True)
        traceback.print_exc()
        raise
