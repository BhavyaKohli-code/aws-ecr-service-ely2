#!/usr/bin/env bash
# Attach the ELY policy engine (created by `agentcore deploy`) to the existing quick-start gateway.
# One-time step; re-run to change the mode.
#
#   bash policy/attach_to_gateway.sh            # LOG_ONLY: records decisions, blocks nothing
#   bash policy/attach_to_gateway.sh ENFORCE    # blocks tool calls the policies do not allow
#   bash policy/attach_to_gateway.sh DETACH     # removes the policy engine from the gateway
set -euo pipefail

MODE="${1:-LOG_ONLY}"
REGION=ap-south-1
GATEWAY_ID=gateway-quick-start-5911f1-51wfgctwi8
WORK=$(mktemp -d)

ENGINE_ARN=$(aws bedrock-agentcore-control list-policy-engines --region "$REGION" \
  --query "policyEngines[?name=='ely_elyPolicyEngine'].policyEngineArn | [0]" --output text)
if [ "$MODE" != "DETACH" ] && { [ -z "$ENGINE_ARN" ] || [ "$ENGINE_ARN" = "None" ]; }; then
  echo "Policy engine ely_elyPolicyEngine not found. Run 'agentcore deploy' first." >&2
  exit 1
fi

aws bedrock-agentcore-control get-gateway --region "$REGION" --gateway-identifier "$GATEWAY_ID" > "$WORK/gateway.json"

# Build the UpdateGateway request from the current gateway, so every other setting is preserved,
# and the IAM permissions the gateway role needs to evaluate policies.
python3 - "$WORK" "$MODE" "$ENGINE_ARN" <<'EOF'
import json, sys
work, mode, engine_arn = sys.argv[1:4]
gw = json.load(open(f"{work}/gateway.json"))
fields = ["name", "description", "roleArn", "protocolType", "protocolConfiguration", "authorizerType",
          "authorizerConfiguration", "kmsKeyArn", "customTransformConfiguration",
          "interceptorConfigurations", "exceptionLevel", "wafConfiguration"]
update = {"gatewayIdentifier": gw["gatewayId"], **{k: gw[k] for k in fields if gw.get(k) is not None}}
if mode != "DETACH":
    update["policyEngineConfiguration"] = {"arn": engine_arn, "mode": mode}
json.dump(update, open(f"{work}/update.json", "w"))
iam = {"Version": "2012-10-17", "Statement": [
    {"Effect": "Allow", "Action": ["bedrock-agentcore:GetPolicyEngine"], "Resource": [engine_arn]},
    {"Effect": "Allow", "Action": ["bedrock-agentcore:AuthorizeAction", "bedrock-agentcore:PartiallyAuthorizeActions"],
     "Resource": [engine_arn, gw["gatewayArn"]]}]}
json.dump(iam, open(f"{work}/iam.json", "w"))
open(f"{work}/role", "w").write(gw["roleArn"].split("/")[-1])
EOF

if [ "$MODE" != "DETACH" ]; then
  ROLE=$(cat "$WORK/role")
  aws iam put-role-policy --role-name "$ROLE" --policy-name ElyPolicyEngineAccess \
    --policy-document "file://$WORK/iam.json"
  echo "Gateway role $ROLE: policy engine permissions granted"
  sleep 10  # let IAM propagate before the gateway checks it
fi

aws bedrock-agentcore-control update-gateway --region "$REGION" --cli-input-json "file://$WORK/update.json" > /dev/null
for _ in $(seq 1 30); do
  STATUS=$(aws bedrock-agentcore-control get-gateway --region "$REGION" --gateway-identifier "$GATEWAY_ID" --query status --output text)
  [ "$STATUS" = "UPDATING" ] || break
  sleep 5
done
echo "Gateway status: $STATUS"
aws bedrock-agentcore-control get-gateway --region "$REGION" --gateway-identifier "$GATEWAY_ID" \
  --query '{policyEngine:policyEngineConfiguration,statusReasons:statusReasons}' --output json
rm -rf "$WORK"
