import os

from strands.models.bedrock import BedrockModel

BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "mistral.ministral-3-8b-instruct")


def load_model() -> BedrockModel:
    """Get Bedrock model client using IAM credentials."""
    # streaming=False as in the original ELY runtime: tool use with Mistral is
    # not reliable over ConverseStream. The agent still streams events to the caller.
    return BedrockModel(model_id=BEDROCK_MODEL_ID, temperature=0.0, streaming=False)
