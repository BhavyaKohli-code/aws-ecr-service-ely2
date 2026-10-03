"""Which sales channel's documents a user may see, read from their Cognito groups.

Some departments keep a folder per channel (s3://<bucket>/DMS_copilot/Sales/uw/agency/..., wpc/ybl/...) next to
files for every channel at the department level. A user in a channel group (CHANNEL_AGENCY, CHANNEL_AXIS,
CHANNEL_DSF, CHANNEL_YBL) gets their own channel's folders and everything that isn't in a channel folder; the other
channels' folders are left out of every search. A user in no channel group sees every channel.

The groups come from the caller's Cognito access token. Runtime's JWT authorizer has already validated that token
(signature, issuer, client) before the agent runs, so its claims can be read here without checking it again.
"""
import base64
import json

CHANNEL_GROUP_PREFIX = "CHANNEL_"
# Folder names that are channels, in any letter case (uw/YBL and wpc/ybl are the same channel)
CHANNELS = {"agency", "axis", "dsf", "ybl"}
CHANNEL_LABELS = {"agency": "Agency", "axis": "Axis", "dsf": "DSF", "ybl": "YBL"}


def _claims(authorization: str | None) -> dict:
    token = (authorization or "").removeprefix("Bearer ").strip()
    try:
        payload = token.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError):
        return {}


def user_channel(authorization: str | None) -> str | None:
    """The caller's channel ("agency", "axis", "dsf" or "ybl"), or None for a user in no channel group."""
    groups = _claims(authorization).get("cognito:groups") or []
    for group in groups if isinstance(groups, list) else []:
        if isinstance(group, str) and group.startswith(CHANNEL_GROUP_PREFIX):
            channel = group[len(CHANNEL_GROUP_PREFIX):].lower()
            if channel in CHANNELS:
                return channel
    return None


def channel_of(uri: str) -> str | None:
    """The channel folder a document is in (s3://bucket/DMS_copilot/<KB>/<department>/<channel>/...), or None
    for a document that isn't in one (department level, or a department without channel folders)."""
    parts = uri.removeprefix("s3://").split("/")  # bucket, DMS_copilot, <KB>, <department>, <folder>, ..., file
    if len(parts) > 5 and parts[4].lower() in CHANNELS:
        return parts[4].lower()
    return None


def visible_to(uri: str, channel: str | None) -> bool:
    """True if a user in `channel` (None: every channel) may see this document."""
    found = channel_of(uri)
    return channel is None or found is None or found == channel


def channel_label(channel: str) -> str:
    return CHANNEL_LABELS.get(channel, channel.upper())
