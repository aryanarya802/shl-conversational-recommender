# SHL Conversational Assessment Recommender

A conversational AI agent that guides hiring managers from a vague intent to a precise shortlist of SHL assessments through multi-turn dialogue.

## Run & Operate

- **Start the API server:** Workflow `artifacts/api-server: API Server` (runs automatically)
- **Health check:** `curl http://localhost:80/api/health`
- **Chat endpoint:** `POST http://localhost:80/api/chat`

## Stack

- Python 3.11, FastAPI, Uvicorn
- OpenAI via Replit AI Integrations proxy (gpt-4.1)
- SHL Product Catalog: 377 items in `artifacts/api-server/catalog.json`
- Keyword-based TF-IDF retrieval in `artifacts/api-server/retrieval.py`

## Where things live

- `artifacts/api-server/main.py` — FastAPI app, endpoints, LLM integration
- `artifacts/api-server/retrieval.py` — Catalog loading and keyword-based retrieval
- `artifacts/api-server/catalog.json` — 377 SHL assessments (entity_id, name, url, test_type, etc.)
- `artifacts/api-server/requirements.txt` — Python dependencies

## API Contract

### GET /api/health
```json
{"status": "ok"}
```

### POST /api/chat
**Request:**
```json
{"messages": [{"role": "user", "content": "..."}]}
```

**Response (strict schema — non-negotiable):**
```json
{
  "reply": "<assistant reply>",
  "recommendations": [
    {"name": "...", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

## Architecture decisions

- **Python FastAPI** replaces the default Node.js api-server because the assignment explicitly requires FastAPI.
- **Stateless design**: Every POST /chat carries the full conversation history; no server-side session state.
- **Retrieval-augmented prompting**: TF-IDF keyword retrieval selects the top 35 catalog items most relevant to the conversation, injected into the system prompt — keeps context window manageable while maximizing recall.
- **Strict output validation**: LLM recommendations are post-filtered against the catalog by exact name match; any hallucinated items are silently dropped before the response is returned.
- **Turn limit enforcement**: Hardcoded at 8 total turns; the LLM is warned at turn 7, and `end_of_conversation` is forced to `true` at turn 8 regardless of LLM output.
- **JSON mode**: OpenAI `response_format: json_object` ensures well-formed JSON on every call.

## Product

The agent:
- Asks clarifying questions when queries are vague (role, seniority, tech stack, language, purpose)
- Recommends 1–10 SHL assessments from the official catalog only
- Accepts refinements and corrections mid-conversation
- Compares assessments when asked, using catalog data
- Politely refuses off-topic requests (legal, general HR)
- Enforces an 8-turn hard cap per conversation

## User preferences

- Python FastAPI backend (not Node.js)
- Deployment target: Replit (primary), also Vercel-compatible structure if needed

## Gotchas

- The system prompt uses Python `.format()` — all literal `{` `}` in the prompt must be escaped as `{{ }}` except the `{catalog}` placeholder.
- The OpenAI client uses Replit AI Integrations env vars: `AI_INTEGRATIONS_OPENAI_BASE_URL` and `AI_INTEGRATIONS_OPENAI_API_KEY`.
- Workflow dev command uses absolute path `/home/runner/workspace/artifacts/api-server` because the runner cwd is not predictable.
- `catalog.json` uses `"url"` (not `"link"`) — normalized from the raw catalog during processing.

## Pointers

- See `pnpm-workspace` skill for workspace structure
- SHL catalog source: `https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json`
