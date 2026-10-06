// Sales digital-tool videos knowledge base: full-time employees and business partners
// (Cognito groups FULL_TIME or BUSINESS_PARTNER), the same people who can search Sales.
permit(
  principal is AgentCore::OAuthUser,
  action == AgentCore::Action::"video-knowledge-retrieval___retrieve_videos",
  resource == AgentCore::Gateway::"arn:aws:bedrock-agentcore:ap-south-1:789270642392:gateway/gateway-quick-start-5911f1-51wfgctwi8"
)
when {
  principal.hasTag("cognito:groups") &&
  (principal.getTag("cognito:groups") like "*FULL_TIME*" ||
   principal.getTag("cognito:groups") like "*BUSINESS_PARTNER*")
};
