"""
SHL Conversational Assessment Recommender — FastAPI service
Endpoints:
  GET  /api/health  → {"status": "ok"}
  POST /api/chat    → {"reply": str, "recommendations": [...], "end_of_conversation": bool}
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from pydantic import BaseModel, field_validator

from retrieval import format_catalog_for_prompt, load_catalog, retrieve

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SHL Conversational Recommender",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load catalog on startup
# ---------------------------------------------------------------------------
_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")


@app.on_event("startup")
def startup_event() -> None:
    load_catalog(_CATALOG_PATH)
    logger.info("SHL catalog loaded.")


# ---------------------------------------------------------------------------
# OpenAI client  (uses Replit AI Integrations proxy)
# ---------------------------------------------------------------------------
_openai_base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
_openai_api_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY", "placeholder")

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-flash"
)

MODEL = "gpt-4.1"  # fast + capable, fits 30-second timeout

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def at_least_one(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages must not be empty")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an SHL assessment advisor. Your job is to guide hiring managers and HR professionals \
from a vague hiring intent to a precise shortlist of SHL assessments through conversation.

HARD RULES (never violate these):
1. Only recommend assessments that appear in the SHL CATALOG provided below. Never invent names or URLs.
2. Return 1–10 recommendations when you have enough context; return an empty list when still gathering info.
3. Each conversation is capped at 8 total turns (user + assistant combined). If turn count reaches 8, \
   set end_of_conversation to true and provide your best shortlist.
4. Set end_of_conversation to true only when the user confirms they are satisfied, or turn limit is reached.
5. Politely refuse any request unrelated to SHL assessments (legal advice, general HR advice, \
   off-topic questions). Say you can only help with SHL assessments.
6. Never hallucinate: don't invent facts, test durations, or URLs not in the catalog.
7. If the query is too vague on turn 1 (e.g. "I need an assessment"), ask clarifying questions \
   instead of recommending immediately.

CONVERSATIONAL BEHAVIORS:
- Ask clarifying questions when needed (role, seniority, skill area, language requirements, purpose).
- Accept refinements mid-conversation ("actually, add personality tests", "drop the Java test").
- Support comparisons between assessments using only catalog data.
- Honor user corrections and update the shortlist accordingly.
- Proactively suggest personality (OPQ32r) and cognitive (Verify G+) tests when appropriate for senior or professional roles.

RESPONSE FORMAT — respond ONLY with valid JSON (no markdown, no explanation outside JSON):
{{
  "reply": "<conversational reply to the user>",
  "recommendations": [
    {{"name": "<exact name from catalog>", "url": "<exact url from catalog>", "test_type": "<exact test_type from catalog>"}},
    ...
  ],
  "end_of_conversation": false
}}
- "recommendations" is [] when still clarifying or refusing off-topic requests.
- "recommendations" has 1-10 items when committing to a shortlist.
- "end_of_conversation" is true only when conversation is complete or turn limit reached.

SHL CATALOG (relevant items for this conversation):
{catalog}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MAX_TURNS = 8


def _count_turns(messages: list[Message]) -> int:
    return len(messages)


def _build_query_from_messages(messages: list[Message]) -> str:
    """Extract all user text to build a retrieval query."""
    return " ".join(m.content for m in messages if m.role == "user")


def _extract_json_from_text(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code fences."""
    text = text.strip()
    # Remove markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    return json.loads(text)


def _validate_and_clean_response(raw: dict, catalog_names: set[str]) -> ChatResponse:
    """Validate LLM response and filter out any non-catalog items."""
    reply = str(raw.get("reply", "")).strip()
    end_conv = bool(raw.get("end_of_conversation", False))
    recs_raw = raw.get("recommendations", []) or []

    validated_recs: list[Recommendation] = []
    for r in recs_raw:
        name = str(r.get("name", "")).strip()
        url = str(r.get("url", "")).strip()
        test_type = str(r.get("test_type", "")).strip()
        # Only include items whose name appears in our catalog
        if name in catalog_names and url.startswith("https://www.shl.com/"):
            validated_recs.append(Recommendation(name=name, url=url, test_type=test_type))
        else:
            logger.warning("Filtered out non-catalog item: %s", name)

    # Cap at 10
    validated_recs = validated_recs[:10]

    return ChatResponse(reply=reply, recommendations=validated_recs, end_of_conversation=end_conv)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    messages = request.messages
    turn_count = _count_turns(messages)

    # Build retrieval query from full conversation
    query = _build_query_from_messages(messages)

    # Retrieve relevant catalog items
    relevant_items = retrieve(query, top_k=35)
    catalog_section = format_catalog_for_prompt(relevant_items)
    catalog_names = {item["name"] for item in relevant_items}

    # Build system prompt with injected catalog
    system_content = SYSTEM_PROMPT.format(catalog=catalog_section)

    # Add turn-limit notice when close to limit
    if turn_count >= MAX_TURNS - 1:
        system_content += (
            f"\n\nIMPORTANT: This is turn {turn_count} of {MAX_TURNS}. "
            "You MUST provide your final shortlist now and set end_of_conversation to true."
        )

    # Build messages for OpenAI
    openai_messages = [{"role": "system", "content": system_content}]
    for msg in messages:
        openai_messages.append({"role": msg.role, "content": msg.content})

    try:
        prompt = system_content + "\n\n" + str(openai_messages)

        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

        response = model.generate_content(prompt)
        
        raw_text = response.text

    except Exception as exc:
        logger.error("Gemini call failed: %s", exc)
        raise HTTPException(status_code=502, detail="LLM service unavailable") from exc

    # Parse and validate
    try:
        raw_json = _extract_json_from_text(raw_text)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Failed to parse LLM JSON: %s | raw=%s", exc, raw_text[:500])
        # Return a safe fallback
        return ChatResponse(
            reply="I encountered an issue processing your request. Could you please rephrase?",
            recommendations=[],
            end_of_conversation=False,
        )

    response = _validate_and_clean_response(raw_json, catalog_names)

    # Hard-enforce turn limit
    if turn_count >= MAX_TURNS:
        response.end_of_conversation = True

    return response
