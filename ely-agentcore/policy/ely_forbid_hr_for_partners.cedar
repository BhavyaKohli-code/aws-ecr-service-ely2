// Business partners can never use the HR knowledge base, even if a broader
// permit policy is added later (forbid always wins in Cedar).
forbid(
  principal is AgentCore::OAuthUser,
  action == AgentCore::Action::"hr-knowledge-retrieval___retrieve_hr",
  resource == AgentCore::Gateway::"arn:aws:bedrock-agentcore:ap-south-1:789270642392:gateway/gateway-quick-start-5911f1-51wfgctwi8"
)
when {
  principal.hasTag("cognito:groups") &&
  principal.getTag("cognito:groups") like "*BUSINESS_PARTNER*"
};
