import json
import logging
import os
from pathlib import PurePosixPath
from urllib.parse import quote

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

# How long the document links stay valid
PRESIGNED_URL_EXPIRY_SECONDS = int(os.getenv("PRESIGNED_URL_EXPIRY_SECONDS", "900"))
# Retrieval returns ~5 chunks per call; only the best-scoring documents are shown as sources
MAX_SOURCES = int(os.getenv("MAX_SOURCES", "3"))
S3_REGION = os.getenv("DOCUMENTS_BUCKET_REGION") or os.getenv("AWS_REGION")

_s3 = None

def _s3_client():
    global _s3
    if _s3 is None:
        # Regional endpoint: the global one redirects for non-us-east-1 buckets, which
        # breaks the signature of a pre-signed URL.
        _s3 = boto3.client(
            "s3",
            region_name=S3_REGION,
            endpoint_url=f"https://s3.{S3_REGION}.amazonaws.com",
            config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
        )
    return _s3


def iter_kb_results(tool_result: dict):
    """Yield Knowledge Base results from a Gateway retrieval tool result."""
    structured = tool_result.get("structuredContent")
    if isinstance(structured, dict) and isinstance(structured.get("results"), list):
        yield from structured["results"]
        return
    for block in tool_result.get("content", []):
        text = block.get("text") if isinstance(block, dict) else None
        if not text:
            continue
        try:
            payload = json.loads(text)
        except ValueError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            yield from payload["results"]


def select_sources(results: list[dict]) -> list[dict]:
    """The distinct S3 documents among knowledge base results.

    One entry per document, keeping its best-scoring chunk, ordered by score. Documents are matched by
    file name, so a file stored in both buckets is listed once.
    """
    by_name: dict[str, dict] = {}
    for result in results:
        uri = (result.get("location") or {}).get("s3Location", {}).get("uri")
        if not uri or not uri.startswith("s3://"):
            continue
        name = PurePosixPath(uri).name.lower()
        score = result.get("score") or 0.0
        if name in by_name and by_name[name]["score"] >= score:
            continue
        text = ((result.get("content") or {}).get("text") or "")
        by_name[name] = {"s3_uri": uri, "score": score, "snippet": " ".join(text.split())[:500],
                         "knowledge_base": result.get("knowledge_base")}
    return sorted(by_name.values(), key=lambda s: s["score"], reverse=True)[:MAX_SOURCES]


def presign_sources(sources: list[dict]) -> list[dict]:
    """Add a document title plus short-lived view and download links to each source."""
    presigned = []
    for source in sources:
        bucket, _, key = source["s3_uri"][len("s3://"):].partition("/")
        filename = PurePosixPath(key).name
        entry = {
            "title": PurePosixPath(filename).stem,
            "file_name": filename,
            **source,
            "expires_in_seconds": PRESIGNED_URL_EXPIRY_SECONDS,
        }
        try:
            s3 = _s3_client()
            quoted = quote(filename)
            entry["view_url"] = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key,
                        "ResponseContentDisposition": f"inline; filename*=UTF-8''{quoted}"},
                ExpiresIn=PRESIGNED_URL_EXPIRY_SECONDS,
            )
            entry["download_url"] = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key,
                        "ResponseContentDisposition": f"attachment; filename*=UTF-8''{quoted}"},
                ExpiresIn=PRESIGNED_URL_EXPIRY_SECONDS,
            )
        except Exception:
            logger.exception("Could not presign %s", source["s3_uri"])
        presigned.append(entry)
    return presigned
