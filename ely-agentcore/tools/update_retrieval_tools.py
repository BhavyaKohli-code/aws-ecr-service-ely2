"""Deploy lambdas/retrieve to both retrieval Lambdas and publish their Gateway tool schemas.

    python tools/update_retrieval_tools.py            # dry run: show what would change
    python tools/update_retrieval_tools.py --apply    # deploy

The Gateway tools and Lambdas live outside the CDK project (quick-start gateway), so this script is how they
are kept in sync with the repo. Tool names are unchanged, so the Cedar access policies are unaffected.
"""
import argparse
import copy
import io
import json
import time
import zipfile
from pathlib import Path

import boto3

REGION = "ap-south-1"
GATEWAY_ID = "gateway-quick-start-5911f1-51wfgctwi8"
LAMBDA_SOURCE = Path(__file__).resolve().parents[1] / "lambdas" / "retrieve" / "lambda_function.py"
TOOLS = {  # gateway target name -> (lambda function, knowledge base id)
    "hr-knowledge-retrieval": ("new-ely2-poc-bhavya-retrievehr", "1SPV0BK1YF"),
    "sales-knowledge-retrieval": ("new-ely2-poc-bhavya-retrievesales", "HOLR19M08V"),
}
EXTRA_INPUTS = {
    "filter": {"type": "string", "description": "Optional Bedrock knowledge base retrieval filter as a JSON string."},
    "number_of_results": {"type": "integer", "description": "Optional number of passages to return (1-25, default 5)."},
}


def lambda_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("lambda_function.py", LAMBDA_SOURCE.read_text(encoding="utf-8"))
    return buf.getvalue()


def main():
    apply = argparse.ArgumentParser()
    apply.add_argument("--apply", action="store_true")
    apply = apply.parse_args().apply
    lam = boto3.client("lambda", region_name=REGION)
    gw = boto3.client("bedrock-agentcore-control", region_name=REGION)
    targets = {t["name"]: t["targetId"] for t in gw.list_gateway_targets(gatewayIdentifier=GATEWAY_ID)["items"]}

    for name, (function, kb_id) in TOOLS.items():
        print(f"{name}: Lambda {function} <- {LAMBDA_SOURCE.name} (KNOWLEDGE_BASE_ID={kb_id}, timeout 15s)")
        if apply:
            lam.update_function_code(FunctionName=function, ZipFile=lambda_zip())
            lam.get_waiter("function_updated_v2").wait(FunctionName=function)
            lam.update_function_configuration(FunctionName=function, Timeout=15,
                                              Environment={"Variables": {"KNOWLEDGE_BASE_ID": kb_id}})
            lam.get_waiter("function_updated_v2").wait(FunctionName=function)

        target = gw.get_gateway_target(gatewayIdentifier=GATEWAY_ID, targetId=targets[name])
        config = copy.deepcopy(target["targetConfiguration"])
        for tool in config["mcp"]["lambda"]["toolSchema"]["inlinePayload"]:
            tool["inputSchema"]["properties"].update(EXTRA_INPUTS)
        print(f"{name}: tool inputs -> {list(config['mcp']['lambda']['toolSchema']['inlinePayload'][0]['inputSchema']['properties'])}")
        if apply:
            gw.update_gateway_target(gatewayIdentifier=GATEWAY_ID, targetId=targets[name], name=name,
                                     targetConfiguration=config,
                                     credentialProviderConfigurations=target["credentialProviderConfigurations"])
            while gw.get_gateway_target(gatewayIdentifier=GATEWAY_ID, targetId=targets[name])["status"] != "READY":
                time.sleep(5)
    print("applied" if apply else "dry run - re-run with --apply")


if __name__ == "__main__":
    main()
