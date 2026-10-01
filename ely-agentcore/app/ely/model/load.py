import os

from strands.models.bedrock import BedrockModel

BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "in.anthropic.claude-sonnet-5")


def accepts_temperature(model_id: str) -> bool:
    """The Claude 5 models reject a `temperature` setting; other models are run at 0."""
    return not any(name in model_id for name in ("claude-sonnet-5", "claude-opus-5", "claude-fable-5"))


def load_model() -> BedrockModel:
    """Get Bedrock model client using IAM credentials."""
    # streaming=False: the whole reply arrives as one text delta, which main.py relies on to remove the
    # trailing USED_PASSAGES line before it reaches the caller. The agent still streams events to the caller.
    settings = {"temperature": 0.0} if accepts_temperature(BEDROCK_MODEL_ID) else {}
    return BedrockModel(model_id=BEDROCK_MODEL_ID, streaming=False, **settings)
