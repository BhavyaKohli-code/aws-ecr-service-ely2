import asyncio
import logging
import os
import uuid

from strands.tools.mcp.mcp_client import MCPClient

from mcp_client.contexts import folder_of
from sources.presign import iter_kb_results

logger = logging.getLogger(__name__)

# Passages given to the model after merging all knowledge bases, best score first
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "8"))
# Passages fetched from each knowledge base. More than the model gets, so the choice of which folders
# to offer sees past the near-identical channel copies (agency/axis/dsf) that fill the top few
SEARCH_RESULTS_PER_KB = int(os.getenv("SEARCH_RESULTS_PER_KB", "12"))
# Passages fetched from a knowledge base when the search is narrowed to some of its folders; the
# retrieval tool can't filter by folder, so it fetches its maximum and the folder is picked here
CONTEXT_SEARCH_RESULTS = 25


def knowledge_base_name(tool_name: str) -> str:
    """Gateway tool names are '<target>___<tool>'; the target names the knowledge base."""
    return tool_name.split("___")[0]


class KnowledgeSearch:
    """Searches every knowledge base the user may use, in parallel, and merges the results by score.

    The Gateway's policy engine filters tools/list per user, so the set searched is exactly what the
    caller is allowed to use (e.g. Sales only for a business partner).
    """

    def __init__(self, mcp_client: MCPClient):
        self._client = mcp_client
        self.tool_names = [
            t.mcp_tool.name for t in mcp_client.list_tools_sync()
            if "query" in (t.mcp_tool.inputSchema or {}).get("properties", {})
        ]
        logger.info("Knowledge search covers: %s", self.tool_names)

    @property
    def knowledge_bases(self) -> list[str]:
        return [knowledge_base_name(n) for n in self.tool_names]

    async def search(self, query: str, context: dict | None = None) -> list[dict]:
        """Return the results across all accessible knowledge bases, best first, each labelled with its
        knowledge base. The caller gives the model only the first MAX_SEARCH_RESULTS.

        With a context ({knowledge_base, folders}) only that knowledge base is searched, and only passages
        from those folders are kept. A knowledge base the user may not use is never searched.
        """
        names, arguments = self.tool_names, {"query": query, "number_of_results": SEARCH_RESULTS_PER_KB}
        if context:
            names = [n for n in self.tool_names if knowledge_base_name(n) == context["knowledge_base"]]
            arguments["number_of_results"] = CONTEXT_SEARCH_RESULTS
        outcomes = await asyncio.gather(
            *(self._client.call_tool_async(f"kb-{uuid.uuid4().hex[:12]}", name, arguments)
              for name in names),
            return_exceptions=True,
        )
        results, failed = [], []
        for name, outcome in zip(names, outcomes):
            if isinstance(outcome, BaseException) or outcome.get("status") == "error":
                logger.warning("Search on %s failed: %s", name, outcome)
                failed.append(name)
                continue
            for result in iter_kb_results(outcome):
                uri = ((result.get("location") or {}).get("s3Location") or {}).get("uri") or ""
                if context and folder_of(uri) not in context["folders"]:
                    continue
                results.append({**result, "knowledge_base": knowledge_base_name(name)})
        if failed and not results:
            raise RuntimeError(f"Knowledge base search failed: {', '.join(failed)}")
        results.sort(key=lambda r: r.get("score") or 0.0, reverse=True)
        return results


def format_passages(results: list[dict]) -> str:
    """Render search results as numbered passages for the model's context."""
    blocks = []
    for i, r in enumerate(results, start=1):
        uri = ((r.get("location") or {}).get("s3Location") or {}).get("uri") or ""
        document = uri.rsplit("/", 1)[-1] or "unknown document"
        text = " ".join(((r.get("content") or {}).get("text") or "").split())
        blocks.append(f"[{i}] Knowledge base: {r.get('knowledge_base')} | Document: {document}\n{text}")
    return "\n\n".join(blocks)
