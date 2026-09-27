"""Knowledge base retrieval tool behind the AgentCore Gateway (one code base for the HR and Sales tools).

Input (MCP tool arguments):
  query              required  text to search for
  filter             optional  Bedrock KB retrieval filter, as a JSON string (built by the agent from taxonomy.json)
  number_of_results  optional  default 5, max 25
Environment:
  KNOWLEDGE_BASE_ID  the knowledge base this tool searches
"""
import json
import os

import boto3

client = boto3.client("bedrock-agent-runtime")
KNOWLEDGE_BASE_ID = os.environ["KNOWLEDGE_BASE_ID"]


def lambda_handler(event, context):
    search = {"numberOfResults": max(1, min(int(event.get("number_of_results") or 5), 25))}
    if event.get("filter"):
        search["filter"] = json.loads(event["filter"]) if isinstance(event["filter"], str) else event["filter"]

    response = client.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": event["query"]},
        retrievalConfiguration={"vectorSearchConfiguration": search},
    )
    return {"results": response["retrievalResults"]}
