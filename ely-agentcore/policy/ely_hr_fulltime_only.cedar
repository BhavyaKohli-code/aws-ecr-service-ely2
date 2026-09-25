// HR knowledge base: full-time employees only (Cognito group FULL_TIME).
// The Gateway exposes the access token's cognito:groups claim as a String tag.
permit(
  principal is AgentCore::OAuthUser,
  action == AgentCore::Action::"hr-knowledge-retrieval___retrieve_hr",
  resource == AgentCore::Gateway::"arn:aws:bedrock-agentcore:ap-south-1:789270642392:gateway/gateway-quick-start-5911f1-51wfgctwi8"
)
when {
  principal.hasTag("cognito:groups") &&
  principal.getTag("cognito:groups") like "*FULL_TIME*"
};
