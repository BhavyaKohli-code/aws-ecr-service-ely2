# ELY Agent API — Integration Guide

This guide is for app developers integrating with ELY, the enterprise employee assistant. ELY answers HR and
Sales questions from company documents and returns the **source documents** with short-lived links to view or
download them.

Every value in this guide was checked against the deployed agent in `ap-south-1`.

---

## 1. How it works

```
 Your app/backend ──(1) username + password──▶ Amazon Cognito
        ▲                                          │
        │◀──────────── AccessToken (1 hour) ───────┘
        │
        └──(2) POST /invocations  + Bearer AccessToken──▶ ELY Agent (AgentCore Runtime)
                                                              │  uses the same token to call
                                                              ▼
                                                   AgentCore Gateway ─▶ HR / Sales Knowledge Bases
        ◀──(3) streamed answer + final event { answer, sources[ view_url, download_url ] }
```

1. Sign the user in with Cognito and get an **AccessToken**.
2. Call the ELY endpoint with that token.
3. Read the streamed response. The last event holds the full answer and its source documents.

---

## 2. Configuration values

| Name | Value |
|---|---|
| AWS region | `ap-south-1` |
| Cognito User Pool ID | `ap-south-1_i4q7zoJr3` |
| Cognito App Client ID | `29v9nta55qtfsbgrupp841900q` |
| Cognito App Client Secret | *Shared separately. Never put it in a mobile or browser app* |
| ELY Runtime ARN | `arn:aws:bedrock-agentcore:ap-south-1:789270642392:runtime/ely_ely-1jeUZ8Ck5s` |
| ELY endpoint | `https://bedrock-agentcore.ap-south-1.amazonaws.com/runtimes/arn%3Aaws%3Abedrock-agentcore%3Aap-south-1%3A789270642392%3Aruntime%2Fely_ely-1jeUZ8Ck5s/invocations?qualifier=DEFAULT` |

> The Runtime ARN in the URL must be URL-encoded (`:` → `%3A`, `/` → `%2F`), as in the endpoint above.

> **Recommended architecture:** the Cognito app client has a **client secret**, so sign-in must run on a server
> you control (a backend-for-frontend). The mobile or web app talks to your backend. The backend gets the token
> from Cognito and calls ELY. This also avoids browser CORS problems when calling the AWS endpoints directly.

---

## 3. Step 1 — Get a Cognito token

### 3.1 Compute `SECRET_HASH`

Because the app client has a secret, every Cognito call needs a `SECRET_HASH` **for that user**:

```
SECRET_HASH = Base64( HMAC_SHA256( key = CLIENT_SECRET, message = USERNAME + CLIENT_ID ) )
```

**Node.js**
```js
import crypto from "crypto";
const secretHash = (username, clientId, clientSecret) =>
  crypto.createHmac("sha256", clientSecret).update(username + clientId).digest("base64");
```

**Python**
```python
import base64, hashlib, hmac
def secret_hash(username, client_id, client_secret):
    digest = hmac.new(client_secret.encode(), (username + client_id).encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()
```

**Bash**
```bash
SECRET_HASH=$(printf '%s' "${USERNAME}${CLIENT_ID}" | openssl dgst -sha256 -hmac "$CLIENT_SECRET" -binary | base64)
```

### 3.2 Sign in (`InitiateAuth`, `USER_PASSWORD_AUTH`)

```bash
curl --request POST \
  --url https://cognito-idp.ap-south-1.amazonaws.com/ \
  --header 'Content-Type: application/x-amz-json-1.0' \
  --header 'X-Amz-Target: AWSCognitoIdentityProviderService.InitiateAuth' \
  --data '{
    "AuthFlow": "USER_PASSWORD_AUTH",
    "ClientId": "29v9nta55qtfsbgrupp841900q",
    "AuthParameters": {
      "USERNAME": "<username>",
      "PASSWORD": "<password>",
      "SECRET_HASH": "<secret hash for this username>"
    }
  }'
```

**Response**
```json
{
  "AuthenticationResult": {
    "AccessToken": "eyJraWQiOi...",
    "ExpiresIn": 3600,
    "IdToken": "eyJraWQiOi...",
    "RefreshToken": "eyJjdHkiOi...",
    "TokenType": "Bearer"
  },
  "ChallengeParameters": {}
}
```

- Use **`AccessToken`** to call ELY. Don't use the IdToken.
- The AccessToken expires after **1 hour** (`ExpiresIn: 3600`).
- If the response contains `ChallengeName` instead of `AuthenticationResult`, the user has to finish a challenge
  first, for example `NEW_PASSWORD_REQUIRED` on first login. Handle it with `RespondToAuthChallenge`.

### 3.3 Refresh the token (no password needed)

```bash
curl --request POST \
  --url https://cognito-idp.ap-south-1.amazonaws.com/ \
  --header 'Content-Type: application/x-amz-json-1.0' \
  --header 'X-Amz-Target: AWSCognitoIdentityProviderService.InitiateAuth' \
  --data '{
    "AuthFlow": "REFRESH_TOKEN_AUTH",
    "ClientId": "29v9nta55qtfsbgrupp841900q",
    "AuthParameters": {
      "REFRESH_TOKEN": "<RefreshToken>",
      "SECRET_HASH": "<secret hash for this username>"
    }
  }'
```

This returns a new `AccessToken` and `IdToken`. Refresh shortly before the AccessToken expires, or when ELY
returns `401` or `403`.

---

## 4. Step 2 — Ask ELY a question

### 4.1 Request

`POST {ELY endpoint}`

| Header | Required | Value |
|---|---|---|
| `Authorization` | Yes | `Bearer <AccessToken>` |
| `Content-Type` | Yes | `application/json` |
| `X-Amzn-Bedrock-AgentCore-Runtime-Session-Id` | Recommended | Conversation ID, **at least 33 characters** (see 4.2). If you leave it out, the request still works, but every question starts a new conversation |

**Body**
```json
{ "prompt": "What is the leave policy?" }
```

**curl**
```bash
TOKEN='<AccessToken>'
URL='https://bedrock-agentcore.ap-south-1.amazonaws.com/runtimes/arn%3Aaws%3Abedrock-agentcore%3Aap-south-1%3A789270642392%3Aruntime%2Fely_ely-1jeUZ8Ck5s/invocations?qualifier=DEFAULT'
SESSION="ely-$(uuidgen)"     # e.g. ely-3f0c...; keep it to continue the same conversation

curl -N -X POST "$URL" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: $SESSION" \
  -d '{"prompt": "What is the leave policy?"}'
```

### 4.2 Conversations (session ID)

- **Same session ID means the same conversation.** ELY remembers earlier questions and answers in that session,
  so follow-ups like *"How many of those can I carry forward?"* work.
- **New session ID means a fresh conversation.** Generate one per chat thread, e.g. `"ely-" + uuid()`, which is 40
  characters.
- A session ID belongs to one user. Don't share one across users.
- Conversation history is kept for **30 days**.

---

## 5. Response format

The response is **Server-Sent Events** (`text/event-stream`), with HTTP status `200`. Each event is a line
`data: <JSON>` followed by a blank line.

### 5.1 Event sequence (real example)

```text
data: {"event": {"messageStart": {"role": "assistant"}}}
data: {"event": {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "tooluse_...", "name": "hr-knowledge-retrieval___retrieve_hr"}}}}}
data: {"event": {"contentBlockDelta": {"delta": {"toolUse": {"input": "{\"query\": \"leave policy\"}"}}}}}
data: {"event": {"contentBlockStop": {}}}
data: {"event": {"messageStop": {"stopReason": "tool_use"}}}
data: {"event": {"metadata": {"usage": {...}, "metrics": {...}}}}
data: {"event": {"messageStart": {"role": "assistant"}}}
data: {"event": {"contentBlockDelta": {"delta": {"text": "Here's an overview of the **Leave Policy** ..."}}}}
data: {"event": {"contentBlockStop": {}}}
data: {"event": {"messageStop": {"stopReason": "end_turn"}}}
data: {"event": {"metadata": {"usage": {...}, "metrics": {...}}}}
data: {"type": "final", "answer": "Here's an overview ...", "sources": [ ... ]}
```

| Event | Meaning | What the app should do |
|---|---|---|
| `event.contentBlockStart.start.toolUse` | ELY is searching a knowledge base (`...retrieve_hr` or `...retrieve_sales`) | Optional: show "Searching HR policies…" |
| `event.contentBlockDelta.delta.text` | A piece of answer text | Optional: append to a live preview |
| `event.contentBlockDelta.delta.toolUse` | The search query being sent | Ignore |
| `event.messageStart / messageStop / contentBlockStop / metadata` | Bookkeeping, token usage | Ignore |
| **`type: "final"`** | **The complete answer and its sources. Always the last event** | **Render this** |
| `error` | Something failed (see 5.4) | Show an error / retry |

> **Use the `final` event as the source of truth.** Streamed text can include a short preamble before a search.
> `final.answer` contains only the final answer. Today the answer text arrives in one chunk, not word by word.

### 5.2 The `final` event

```json
{
  "type": "final",
  "answer": "Here's an overview of the **Leave Policy** at Axis Max Life (effective from **1st April 2026**): ...",
  "sources": [
    {
      "title": "Leave and Attendance Policy",
      "file_name": "Leave and Attendance Policy.docx",
      "s3_uri": "s3://elydocuments/HRIS/HR Policies/Leave and Attendance Policy.docx",
      "score": 0.503,
      "snippet": "Leave and Attendance Policy Version Last reviewed 1st April 2026 Version 2.6 effective from 1st April 2026 ...",
      "expires_in_seconds": 900,
      "view_url": "https://elydocuments.s3.ap-south-1.amazonaws.com/HRIS/HR%20Policies/Leave%20and%20Attendance%20Policy.docx?X-Amz-...",
      "download_url": "https://elydocuments.s3.ap-south-1.amazonaws.com/HRIS/HR%20Policies/Leave%20and%20Attendance%20Policy.docx?X-Amz-..."
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `answer` | string | Final answer in **Markdown** (headings, bold, lists, tables). Render it as Markdown |
| `sources` | array | Up to **3** documents the answer was retrieved from, best match first. **Empty** (`[]`) when ELY didn't search, e.g. for small talk |
| `sources[].title` | string | Display name (file name without extension) |
| `sources[].file_name` | string | File name with extension (`.docx`, `.pdf`) |
| `sources[].score` | number | Relevance, 0–1. Higher is more relevant |
| `sources[].snippet` | string | The best-matching passage, up to 500 characters. Use as a preview |
| `sources[].view_url` | string | Pre-signed link that asks the browser to **open** the file (PDFs open in the browser; `.docx` files download) |
| `sources[].download_url` | string | Pre-signed link that always **downloads** the file with its original name |
| `sources[].expires_in_seconds` | number | Link lifetime, currently **900 s (15 min)** from when the answer was generated |
| `sources[].s3_uri` | string | Internal S3 location. For logging only, not for display |

**Suggested UI**

```
┌───────────────────────────────────────────────┐
│ <answer rendered as Markdown>                 │
├───────────────────────────────────────────────┤
│ Sources                                        │
│  📄 Leave and Attendance Policy   [View] [⬇]  │
│     "Leave and Attendance Policy Version 2.6…" │
│  📄 Exit Policy                   [View] [⬇]  │
└───────────────────────────────────────────────┘
```

**Link rules**
- Links expire after 15 minutes. **Don't store them.** To reopen a document later, ask again or add a
  "refresh links" call in your backend.
- The links are anonymous. Anyone who has one can open it until it expires, so don't log them or send them to
  third parties.

### 5.3 Minimal client examples

**JavaScript (fetch, streaming)**
```js
async function askEly({ token, sessionId, prompt, onText }) {
  const res = await fetch(ELY_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": sessionId,
    },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) throw new Error(`ELY HTTP ${res.status}: ${await res.text()}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "", final = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, idx); buffer = buffer.slice(idx + 2);
      if (!chunk.startsWith("data: ")) continue;
      const data = JSON.parse(chunk.slice(6));
      if (data.error) throw new Error(data.error);
      const text = data.event?.contentBlockDelta?.delta?.text;
      if (text && onText) onText(text);
      if (data.type === "final") final = data;
    }
  }
  return final; // { answer, sources }
}
```

**Python (requests)**
```python
import json, requests

def ask_ely(token, session_id, prompt):
    r = requests.post(ELY_URL, stream=True, timeout=180, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
    }, json={"prompt": prompt})
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = json.loads(line[6:])
        if "error" in data:
            raise RuntimeError(data["error"])
        if data.get("type") == "final":
            return data          # {"type": "final", "answer": ..., "sources": [...]}
```

### 5.4 Errors

| What you see | Cause | Fix |
|---|---|---|
| HTTP `403` `{"message":"OAuth authorization failed: ..."}` (or `401`) | Missing, expired, or wrong token (e.g. IdToken instead of AccessToken) | Refresh the token (3.3) and retry |
| HTTP `400` `"...'runtimeSessionId' failed to satisfy constraint: Member must have length greater than or equal to 33"` | Session ID shorter than 33 characters | Use `"ely-" + uuid()` |
| HTTP `200` with `data: {"error": "...", "error_type": "...", "message": "An error occurred during streaming"}` | Failure inside the agent after the stream started | Show a friendly error and retry once. Report persistent errors with the time and session ID |
| Cognito `NotAuthorizedException` | Wrong username or password, or wrong `SECRET_HASH` | Check the credentials. The hash must use the **same username** you send |
| Cognito `UserNotConfirmedException` / `PasswordResetRequiredException` | Account state | Complete the Cognito account flow |

Latency: typically **3–10 s** for a complete answer (search plus generation). Use a client timeout of at least
**120 s**.

---

## 6. What ELY can answer

| Knowledge base | Tool | Covers |
|---|---|---|
| HR | `hr-knowledge-retrieval___retrieve_hr` | HR policies (leave and attendance, exit, recruitment, scholarship, …) |
| Sales | `sales-knowledge-retrieval___retrieve_sales` | Business insurance (Keyman, employer–employee, partnership) and NRI topics |

ELY picks the knowledge base itself; the app just sends the question. If the documents don't contain the
answer, ELY says the information isn't available instead of guessing.

---

## 7. Appendix — Query the knowledge bases directly (debugging)

Use this to see exactly what a knowledge base returns for a query, without the agent. It uses the same
AccessToken.

```bash
GW='https://gateway-quick-start-5911f1-51wfgctwi8.gateway.bedrock-agentcore.ap-south-1.amazonaws.com/mcp'

# List tools
curl -s -X POST "$GW" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" -H "MCP-Protocol-Version: 2025-11-25" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

# Search the HR knowledge base (use sales-knowledge-retrieval___retrieve_sales for Sales)
curl -s -X POST "$GW" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" -H "MCP-Protocol-Version: 2025-11-25" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call",
       "params":{"name":"hr-knowledge-retrieval___retrieve_hr","arguments":{"query":"leave policy"}}}'
```

The result's `content[0].text` is a JSON string `{"results": [ {content.text, score, location.s3Location.uri}, … ]}`
with the top 5 matching passages.
