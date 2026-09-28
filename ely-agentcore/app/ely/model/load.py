import os

from strands.models.bedrock import BedrockModel

BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "openai.gpt-oss-20b-1:0")


def load_model() -> BedrockModel:
    """Get Bedrock model client using IAM credentials."""
    # streaming=False: the whole reply arrives as one text delta, which main.py relies on to remove the
    # trailing USED_PASSAGES line before it reaches the caller. The agent still streams events to the caller.
    return BedrockModel(model_id=BEDROCK_MODEL_ID, temperature=0.0, streaming=False)
