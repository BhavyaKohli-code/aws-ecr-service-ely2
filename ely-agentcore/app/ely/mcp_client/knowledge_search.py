import asyncio
import json
import logging
import uuid

from strands.tools.mcp.mcp_client import MCPClient

from knowledge import taxonomy
from sources.presign import iter_kb_results

logger = logging.getLogger(__name__)


def knowledge_base_name(tool_name: str) -> str:
    """Gateway tool names are '<target>___<tool>'; the target names the knowledge base."""
    return tool_name.split("___")[0]


class KnowledgeSearch:
    """Retrieval steered by taxonomy.json, over the knowledge bases this user may use.

    The Gateway's policy engine filters tools/list per user (e.g. Sales only for a business partner), so
    everything here only ever searches what the caller is allowed to see.

    1. Rules pick the question's preferred category (and any product it names); search only that category.
    2. If no rule applies, or that category has nothing, search everything the user may use.
    Duplicate documents, the duplicate HR data source and self-help reading are always filtered out
    (self-help only when the question isn't about personal development).
    """

    def __init__(self, mcp_client: MCPClient):
        self._client = mcp_client
        self.tools = {  # tool name -> domain of the knowledge base it searches
            t.mcp_tool.name: taxonomy.knowledge_base_domain(knowledge_base_name(t.mcp_tool.name))
            for t in mcp_client.list_tools_sync()
            if "query" in (t.mcp_tool.inputSchema or {}).get("properties", {})
        }
        logger.info("Knowledge search covers: %s", self.tools)

    @property
    def knowledge_bases(self) -> list[str]:
        return [knowledge_base_name(n) for n in self.tools]

    async def search(self, query: str) -> tuple[list[dict], str | None]:
        """Top passages for the query, each labelled with its knowledge base and category; plus the rule's category."""
        settings = taxonomy.search_settings()
        preferred, products = taxonomy.preferred_category(query), taxonomy.detect_products(query)
        base = [{"notEquals": {"key": "duplicate", "value": True}},
                {"notIn": {"key": "x-amz-bedrock-kb-data-source-id", "value": taxonomy.excluded_data_sources()}}]
        if preferred != "hr_self_help":
            base.append({"notEquals": {"key": "category", "value": "hr_self_help"}})

        results = []
        if preferred:
            only = [{"equals": {"key": "category", "value": preferred}}]
            product = [{"in": {"key": "product", "value": products}}] if products and preferred in ("products", "dcc", "uw") else []
            tools = [t for t, d in self.tools.items() if d == taxonomy.domain(preferred)]
            results = await self._search(tools, query, base + only + product, settings["preferred_results"])
            if not results and product:  # the product filter was too strict: category only
                results = await self._search(tools, query, base + only, settings["preferred_results"])
        if not results:
            results = await self._search(list(self.tools), query, base, settings["open_results"])
            sales = [t for t, d in self.tools.items() if d == "SALES"]
            if products and sales:
                results += await self._search(sales, query, base + [{"in": {"key": "product", "value": products}}],
                                              settings["product_results"])
        logger.info("Search: preferred=%s products=%s -> %d passages", preferred, products, len(results))
        return _dedupe(sorted(results, key=lambda r: r.get("score") or 0.0, reverse=True))[:settings["max_passages"]], preferred

    async def _search(self, tools: list[str], query: str, conditions: list[dict], n: int) -> list[dict]:
        if not tools:
            return []
        arguments = {"query": query, "number_of_results": n,
                     "filter": json.dumps({"andAll": conditions} if len(conditions) > 1 else conditions[0])}
        outcomes = await asyncio.gather(
            *(self._client.call_tool_async(f"kb-{uuid.uuid4().hex[:12]}", t, arguments) for t in tools),
            return_exceptions=True,
        )
        results, failed = [], []
        for tool, outcome in zip(tools, outcomes):
            if isinstance(outcome, BaseException) or outcome.get("status") == "error":
                logger.warning("Search on %s failed: %s", tool, outcome)
                failed.append(tool)
                continue
            for result in iter_kb_results(outcome):
                results.append({**result, "knowledge_base": knowledge_base_name(tool),
                                "category": (result.get("metadata") or {}).get("category", "")})
        if failed and len(failed) == len(tools):
            raise RuntimeError(f"Knowledge base search failed: {', '.join(failed)}")
        return results


def _dedupe(results: list[dict]) -> list[dict]:
    """Drop passages whose text repeats an earlier (higher-scoring) one."""
    seen, out = set(), []
    for r in results:
        key = " ".join(((r.get("content") or {}).get("text") or "").split())[:300]
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def format_passages(results: list[dict]) -> str:
    """Render search results as numbered passages, labelled by category, for the model's context."""
    blocks = []
    for i, r in enumerate(results, start=1):
        uri = ((r.get("location") or {}).get("s3Location") or {}).get("uri") or ""
        document = uri.rsplit("/", 1)[-1] or "unknown document"
        text = " ".join(((r.get("content") or {}).get("text") or "").split())
        blocks.append(f"[{i}] Category: {taxonomy.label(r.get('category', ''))} | Document: {document}\n{text}")
    return "\n\n".join(blocks)
