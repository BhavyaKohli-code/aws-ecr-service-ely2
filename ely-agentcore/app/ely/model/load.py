import os

from strands.models.bedrock import BedrockModel

BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "in.anthropic.claude-sonnet-5")


def accepts_temperature(model_id: str) -> bool:
    """The Claude 5 models reject a `temperature` setting; other models are run at 0."""
    return not any(name in model_id for name in ("claude-sonnet-5", "claude-opus-5", "claude-fable-5"))


def load_model() -> BedrockModel:
    """Get Bedrock model client using IAM credentials."""
    # Streamed, so the caller sees the answer as it is written; main.py holds back the markers and the
    # trailing USED_PASSAGES line (see model/reply.py).
    settings = {"temperature": 0.0} if accepts_temperature(BEDROCK_MODEL_ID) else {}
    return BedrockModel(model_id=BEDROCK_MODEL_ID, streaming=True, **settings)
