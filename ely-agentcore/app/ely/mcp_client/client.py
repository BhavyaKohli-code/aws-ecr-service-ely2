import os
import logging
from typing import Optional

import httpx
from mcp.client.streamable_http import streamablehttp_client
from strands.tools.mcp.mcp_client import MCPClient

logger = logging.getLogger(__name__)

# ELY AgentCore Gateway (HR / Sales knowledge retrieval tools)
GATEWAY_URL = os.getenv(
    "GATEWAY_URL",
    "https://gateway-quick-start-5911f1-51wfgctwi8.gateway.bedrock-agentcore.ap-south-1.amazonaws.com/mcp",
)


class GatewayAuth(httpx.Auth):
    """Attaches the caller's current Cognito token to every Gateway request.

    The MCP client lives for the whole runtime session, but the caller's token
    can be refreshed between invocations, so the token is read per request.
    """

    def __init__(self):
        self.authorization: Optional[str] = None

    def auth_flow(self, request):
        if self.authorization:
            request.headers["Authorization"] = self.authorization
        yield request


def get_gateway_mcp_client(auth: GatewayAuth) -> MCPClient:
    """Returns an MCP Client for the ELY Gateway, compatible with Strands"""
    return MCPClient(lambda: streamablehttp_client(GATEWAY_URL, auth=auth))
