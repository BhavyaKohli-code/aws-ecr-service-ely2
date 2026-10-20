"""Web search for public, time-sensitive facts (fund performance, NAV, bonus declared, claim settlement ratio, Nifty),
run in an AgentCore Browser session.

The question's web query (see mcp_client/follow_up.py) is searched on a search engine's results page, and the top
results are opened and read. Their text becomes passages the answering model reads after the knowledge base ones,
each labelled as a web page with its link, and the pages it used are returned as sources.

The browser is driven over the Chrome DevTools Protocol with `websockets` rather than Playwright: the runtime already
ships it, and Playwright would add a Node.js driver and a compiled dependency to the package. Images, fonts and media
are not loaded, and every page gets a time limit, since a slow page (ads, trackers) would otherwise hold up the
answer; a page that doesn't load in time is answered from its search snippet.

Google answers an automated browser with a CAPTCHA, so the default engine is DuckDuckGo's HTML results page. The
engine is the WEB_SEARCH_URL setting; RESULTS_JS reads DuckDuckGo's and Google's results pages.
"""
import asyncio
import json
import logging
import os
import time
import urllib.parse

from bedrock_agentcore.tools.browser_client import BrowserClient
from websockets.asyncio.client import connect

logger = logging.getLogger(__name__)

# The AgentCore Browser to use; no web search without it
BROWSER_ID = os.getenv("BROWSER_ID", "")
REGION = os.getenv("AWS_REGION", "ap-south-1")
# The search engine's results page; {query} is replaced by the URL-encoded query
WEB_SEARCH_URL = os.getenv("WEB_SEARCH_URL", "https://html.duckduckgo.com/html/?q={query}")
# Results whose pages are opened and read; the others are given to the model as their search snippet
WEB_PAGES = int(os.getenv("WEB_PAGES", "2"))
# Results given to the model in all
WEB_RESULTS = int(os.getenv("WEB_RESULTS", "5"))
# Seconds for one page to load and be read, and for the whole search
PAGE_TIMEOUT = float(os.getenv("WEB_PAGE_TIMEOUT_SECONDS", "8"))
WEB_SEARCH_TIMEOUT = float(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "25"))
# Characters of a page's text given to the model
PAGE_CHARS = int(os.getenv("WEB_PAGE_CHARS", "6000"))
# Once a page has loaded, how long its scripts get to fill in figures (fund pages load them after the page)
SETTLE_SECONDS = 2.0

BLOCKED = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.svg", "*.ico", "*.woff", "*.woff2", "*.ttf", "*.otf",
           "*.mp4", "*.webm", "*.mp3", "*doubleclick.net*", "*googlesyndication*", "*google-analytics*",
           "*googletagmanager*", "*facebook.net*", "*hotjar*", "*clarity.ms*"]

# The results on a results page: DuckDuckGo's HTML page, else Google's (a link with a heading)
RESULTS_JS = """(() => {
  let rows = [...document.querySelectorAll('.result')].map(r => {
    const a = r.querySelector('.result__a');
    return a && {title: a.innerText, url: a.href, snippet: (r.querySelector('.result__snippet') || {}).innerText || ''};
  }).filter(Boolean);
  if (!rows.length) rows = [...document.querySelectorAll('a:has(h3)')].map(a => {
    const box = a.closest('div.g, div[data-hveid]') || a.parentElement;
    return {title: a.querySelector('h3').innerText, url: a.href, snippet: (box ? box.innerText : '').slice(0, 400)};
  });
  return rows;
})()"""
TEXT_JS = f"(document.body ? document.body.innerText : '').replace(/\\n\\s*\\n+/g, '\\n').slice(0, {PAGE_CHARS})"


class _Cdp:
    """Commands over one DevTools connection to the browser; replies are matched to commands by id."""

    def __init__(self, ws):
        self._ws = ws
        self._next = 0
        self._waiting: dict[int, asyncio.Future] = {}
        self._reader = asyncio.create_task(self._read())

    async def _read(self):
        try:
            async for raw in self._ws:
                message = json.loads(raw)
                future = self._waiting.pop(message.get("id"), None)
                if future is None or future.done():
                    continue  # an event, or a reply nobody waits for any more
                if "error" in message:
                    future.set_exception(RuntimeError(message["error"].get("message", "DevTools error")))
                else:
                    future.set_result(message.get("result") or {})
        finally:
            for future in self._waiting.values():
                if not future.done():
                    future.set_exception(ConnectionError("The browser connection closed"))

    async def call(self, method: str, params: dict | None = None, session: str | None = None) -> dict:
        self._next += 1
        future = asyncio.get_running_loop().create_future()
        self._waiting[self._next] = future
        message = {"id": self._next, "method": method, "params": params or {}}
        if session:
            message["sessionId"] = session
        await self._ws.send(json.dumps(message))
        return await future

    def close(self):
        self._reader.cancel()


async def _read_page(cdp: _Cdp, url: str, expression: str):
    """Open url in a new tab, wait for it to load, and return the value of expression there."""
    target = (await cdp.call("Target.createTarget", {"url": "about:blank"}))["targetId"]
    try:
        session = (await cdp.call("Target.attachToTarget", {"targetId": target, "flatten": True}))["sessionId"]
        await cdp.call("Network.enable", session=session)
        await cdp.call("Network.setBlockedURLs", {"urls": BLOCKED}, session=session)
        await cdp.call("Page.navigate", {"url": url}, session=session)
        loaded = "location.href !== 'about:blank' && document.readyState !== 'loading'"
        while not (await cdp.call("Runtime.evaluate", {"expression": loaded, "returnByValue": True},
                                  session=session))["result"].get("value"):
            await asyncio.sleep(0.3)
        await asyncio.sleep(SETTLE_SECONDS)
        result = await cdp.call("Runtime.evaluate", {"expression": expression, "returnByValue": True}, session=session)
        return result["result"].get("value")
    finally:
        try:
            await cdp.call("Target.closeTarget", {"targetId": target})
        except Exception:
            pass


def _real_url(url: str) -> str:
    """DuckDuckGo's results link to a redirect (/l/?uddg=<the page>); the page itself."""
    parsed = urllib.parse.urlparse(url)
    target = urllib.parse.parse_qs(parsed.query).get("uddg")
    return target[0] if target else url


def _results(rows) -> list[dict]:
    """Distinct web results, without the engine's own pages and adverts."""
    results, seen = [], set()
    for row in rows or []:
        url = _real_url(row.get("url") or "")
        host = urllib.parse.urlparse(url).netloc.lower()
        if not url.startswith("http") or url in seen or any(h in host for h in ("duckduckgo.", "google.", "bing.")):
            continue
        seen.add(url)
        results.append({"title": " ".join((row.get("title") or "").split()) or host, "url": url,
                        "snippet": " ".join((row.get("snippet") or "").split())})
    return results[:WEB_RESULTS]


async def _search(query: str) -> list[dict]:
    started = time.monotonic()
    client = BrowserClient(REGION)
    await asyncio.to_thread(client.start, identifier=BROWSER_ID, name="ely-web-search", session_timeout_seconds=120)
    try:
        url, headers = client.generate_ws_headers()
        async with connect(url, additional_headers=headers, max_size=None, open_timeout=10) as ws:
            cdp = _Cdp(ws)
            try:
                search_url = WEB_SEARCH_URL.replace("{query}", urllib.parse.quote_plus(query))
                results = _results(await asyncio.wait_for(_read_page(cdp, search_url, RESULTS_JS), PAGE_TIMEOUT))
                # PDFs show no text in the browser: those are given to the model as their snippet
                pages = [r for r in results if not urllib.parse.urlparse(r["url"]).path.lower().endswith(".pdf")]
                pages = pages[:WEB_PAGES]
                texts = await asyncio.gather(
                    *(asyncio.wait_for(_read_page(cdp, r["url"], TEXT_JS), PAGE_TIMEOUT) for r in pages),
                    return_exceptions=True)
                for result, text in zip(pages, texts):
                    if isinstance(text, BaseException):
                        logger.info("Web page %s not read (%s); using its snippet", result["url"], type(text).__name__)
                    elif isinstance(text, str):
                        result["text"] = text.strip()
            finally:
                cdp.close()
        logger.info("Web search for %r: %d result(s), %d page(s) read, %.1fs", query, len(results),
                    sum(1 for r in results if r.get("text")), time.monotonic() - started)
        return results
    finally:
        await asyncio.to_thread(client.stop)


async def web_search(query: str) -> list[dict]:
    """The top web results for query: [{title, url, snippet, text?}], text being the page's own text for the pages
    that were read. Empty when there is no browser configured or the search fails or takes too long, so the
    question is answered from the knowledge base alone."""
    if not BROWSER_ID or not query:
        return []
    try:
        return await asyncio.wait_for(_search(query), WEB_SEARCH_TIMEOUT)
    except Exception as exc:  # including the time limit: the answer goes ahead without the web
        logger.warning("Web search for %r failed: %s: %s", query, type(exc).__name__, exc)
        return []


def as_passages(results: list[dict]) -> list[dict]:
    """Web results in the shape of knowledge base results, to be numbered and given to the model with them."""
    passages = []
    for result in results:
        text = "\n".join(t for t in (result.get("snippet"), result.get("text")) if t)
        if not text:
            continue
        passages.append({"knowledge_base": "web", "content": {"text": text[:PAGE_CHARS]},
                         "location": {"webLocation": {"url": result["url"]}}, "title": result["title"],
                         "snippet": result.get("snippet") or text[:300]})
    return passages


def web_sources(passages: list[dict]) -> list[dict]:
    """The web pages an answer used, as sources: a title, the site and the link (opened in the system browser)."""
    sources = []
    for passage in passages:
        url = passage["location"]["webLocation"]["url"]
        # A public link doesn't expire like a pre-signed one; a day keeps it valid in a chat reopened later
        sources.append({"source_type": "web", "title": passage["title"],
                        "file_name": urllib.parse.urlparse(url).netloc.removeprefix("www."),
                        "url": url, "view_url": url, "expires_in_seconds": 86400,
                        "snippet": passage["snippet"][:500], "knowledge_base": "web"})
    return sources
