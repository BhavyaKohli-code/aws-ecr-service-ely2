import asyncio
import logging
import os
import re
import uuid

from strands.tools.mcp.mcp_client import MCPClient

from mcp_client.channels import visible_to
from mcp_client.contexts import folder_of
from sources.presign import iter_kb_results

logger = logging.getLogger(__name__)

# Passages given to the model after merging all knowledge bases and search texts, best score first. Each
# question is searched several ways (see follow_up.py), so more distinct passages compete for these places
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "12"))
# Passages after those, given to the model in a second look only when the first ones don't answer
SECOND_LOOK_RESULTS = int(os.getenv("SECOND_LOOK_RESULTS", "18"))
# A passage is a repeat when this share of its 5-word sequences is in a better-ranked passage
REPEAT_OVERLAP = 0.8
# Passages fetched from each knowledge base. More than the model gets, so the choice of which folders
# to offer sees past the near-identical channel copies (agency/axis/dsf) that fill the top few
SEARCH_RESULTS_PER_KB = int(os.getenv("SEARCH_RESULTS_PER_KB", "12"))
# Passages fetched from a knowledge base when the search is narrowed to some of its folders or to the user's
# channel; the retrieval tool can't filter by folder, so it fetches its maximum and the folder is picked here
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

    async def search(self, queries: list[str], context: dict | None = None,
                     knowledge_bases: list[str] | None = None, channel: str | None = None) -> list[dict]:
        """Return the results across all accessible knowledge bases, best first, each labelled with its
        knowledge base. The caller gives the model only the first MAX_SEARCH_RESULTS.

        Every query is searched in every knowledge base, all in parallel; a passage found by several queries
        is kept once, with its best score.

        With a context ({knowledge_base, folders}) only that knowledge base is searched, and only passages
        from those folders are kept. With knowledge_bases (a specialist chat) only those are searched. A
        knowledge base the user may not use is never searched. With a channel ("agency", ...) passages from
        other channels' folders are dropped (see channels.py).
        """
        names, number_of_results = self.tool_names, SEARCH_RESULTS_PER_KB
        if knowledge_bases is not None:
            names = [n for n in names if knowledge_base_name(n) in knowledge_bases]
        if context:
            names = [n for n in self.tool_names if knowledge_base_name(n) == context["knowledge_base"]]
            number_of_results = CONTEXT_SEARCH_RESULTS
        if channel:
            number_of_results = CONTEXT_SEARCH_RESULTS  # other channels' passages are dropped below
        calls = [(name, query) for query in queries for name in names]
        outcomes = await asyncio.gather(
            *(self._client.call_tool_async(f"kb-{uuid.uuid4().hex[:12]}", name,
                                           {"query": query, "number_of_results": number_of_results})
              for name, query in calls),
            return_exceptions=True,
        )
        best, failed = {}, []
        for (name, query), outcome in zip(calls, outcomes):
            if isinstance(outcome, BaseException) or outcome.get("status") == "error":
                logger.warning("Search on %s for %r failed: %s", name, query, outcome)
                failed.append(name)
                continue
            for result in iter_kb_results(outcome):
                uri = ((result.get("location") or {}).get("s3Location") or {}).get("uri") or ""
                if context and folder_of(uri) not in context["folders"]:
                    continue
                if not visible_to(uri, channel):
                    continue
                key = (uri, (result.get("content") or {}).get("text") or "")
                if key not in best or (result.get("score") or 0.0) > (best[key].get("score") or 0.0):
                    best[key] = {**result, "knowledge_base": knowledge_base_name(name)}
        if failed and not best:
            raise RuntimeError(f"Knowledge base search failed: {', '.join(dict.fromkeys(failed))}")
        return drop_repeats(sorted(best.values(), key=lambda r: r.get("score") or 0.0, reverse=True))


def _shingles(text: str, size: int = 5) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + size]) for i in range(max(1, len(words) - size + 1))}


def drop_repeats(results: list[dict]) -> list[dict]:
    """Results without passages that repeat a better one: the same file is kept in more than one folder and
    knowledge base (Business Insurance in HR and Sales, the commission FAQ in HR payroll and Sales DCC), and
    a copy would take a place among the passages the model reads and show up as a second area to choose."""
    kept, seen = [], []
    for result in results:
        words = _shingles((result.get("content") or {}).get("text") or "")
        if any(len(words & other) >= REPEAT_OVERLAP * min(len(words), len(other)) for other in seen):
            continue
        kept.append(result)
        seen.append(words)
    return kept


def format_passages(results: list[dict]) -> str:
    """Render search results as numbered passages for the model's context."""
    blocks = []
    for i, r in enumerate(results, start=1):
        uri = ((r.get("location") or {}).get("s3Location") or {}).get("uri") or ""
        document = uri.rsplit("/", 1)[-1] or "unknown document"
        text = " ".join(((r.get("content") or {}).get("text") or "").split())
        blocks.append(f"[{i}] Knowledge base: {r.get('knowledge_base')} | Document: {document}\n{text}")
    return "\n\n".join(blocks)
