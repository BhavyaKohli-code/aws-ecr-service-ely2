"""Which area of the knowledge base a question is about, read from the document folders.

Documents live at s3://<bucket>/DMS_copilot/<HR|Sales>/<folder>/..., and the top-level folder (uw, dcc,
wpc, payroll, ...) is what tells apart answers that match equally well. When good matches come from more
than one folder, ELY asks the user to pick one before answering, then searches only that folder.
"""
import os
import re
from pathlib import PurePosixPath

# A match at or above this score counts towards a folder being offered as a choice
CLARIFY_MIN_SCORE = float(os.getenv("CLARIFY_MIN_SCORE", "0.5"))
# ...and only if the folder's best match is this close to the best match overall. Hierarchical chunking
# scores most passages well above 0.5, so without this almost every question would be asked back.
CLARIFY_MARGIN = float(os.getenv("CLARIFY_MARGIN", "0.10"))
# At most this many choices are offered, best match first
MAX_CONTEXT_OPTIONS = int(os.getenv("MAX_CONTEXT_OPTIONS", "4"))

# Display names for folders; any other folder is shown as its name with underscores as spaces
FOLDER_LABELS = {
    "business_insurance_nri": "Business insurance & NRI",
    "competition": "Competition",
    "dcc": "DCC",
    "digital_tool": "Digital tools",
    "policy_status": "Policy status",
    "products": "Products",
    "uw": "Underwriting",
    "wpc": "WPC grids",
    "employee_services": "Employee services",
    "incentives": "Incentives",
    "leaves": "Leaves",
    "payroll": "Payroll",
}

_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def folder_of(uri: str) -> str | None:
    """The top-level folder under the knowledge base root, or None for a file at the root."""
    parts = uri.removeprefix("s3://").split("/")  # bucket, DMS_copilot, <KB>, <folder>, ..., file
    return parts[3] if len(parts) > 4 else None


def folder_label(folder: str) -> str:
    return FOLDER_LABELS.get(folder.lower(), folder.replace("_", " ").strip().capitalize())


def _uri(result: dict) -> str:
    return ((result.get("location") or {}).get("s3Location") or {}).get("uri") or ""


def context_options(results: list[dict]) -> list[dict]:
    """The folders the good matches come from, as choices: [{knowledge_base, folders, label, score}].

    The same file is stored in more than one folder (HR keeps copies in payroll/ and employee_services/),
    so folders that share a matched document are one choice rather than two that lead to the same answer.
    """
    folders_of_doc: dict[tuple[str, str], set[str]] = {}
    best: dict[tuple[str, str], float] = {}
    for result in results:
        score = result.get("score") or 0.0
        folder = folder_of(_uri(result))
        if score < CLARIFY_MIN_SCORE or not folder:
            continue
        kb = result.get("knowledge_base") or ""
        doc = (kb, PurePosixPath(_uri(result)).name.lower())
        folders_of_doc.setdefault(doc, set()).add(folder)
        best[(kb, folder)] = max(best.get((kb, folder), 0.0), score)

    # Merge folders that hold the same document (per knowledge base)
    groups: list[tuple[str, set[str]]] = []
    for (kb, _), folders in folders_of_doc.items():
        merged = set(folders)
        for group in [g for g in groups if g[0] == kb and g[1] & merged]:
            merged |= group[1]
            groups.remove(group)
        groups.append((kb, merged))

    options = []
    for kb, folders in groups:
        ordered = sorted(folders, key=lambda f: best[(kb, f)], reverse=True)
        options.append({
            "knowledge_base": kb,
            "folders": ordered,
            "label": " / ".join(folder_label(f) for f in ordered),
            "score": round(best[(kb, ordered[0])], 4),
        })
    options.sort(key=lambda o: o["score"], reverse=True)
    if options:
        options = [o for o in options if o["score"] >= options[0]["score"] - CLARIFY_MARGIN]
    return options[:MAX_CONTEXT_OPTIONS]


def parse_context(raw) -> dict | None:
    """The caller's chosen context ({knowledge_base, folders}), validated; None if absent or malformed."""
    if not isinstance(raw, dict):
        return None
    kb, folders = raw.get("knowledge_base"), raw.get("folders")
    if not isinstance(kb, str) or not _NAME.match(kb) or not isinstance(folders, list):
        return None
    folders = [f for f in folders if isinstance(f, str) and _NAME.match(f)][:5]
    return {"knowledge_base": kb, "folders": folders} if folders else None
