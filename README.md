# SHL Conversational Recommender API

A FastAPI-based conversational recommendation system built for the SHL AI Internship Assignment.

## Features

- FastAPI backend with:
  - `GET /api/health`
  - `POST /api/chat`
- Stateless conversational flow
- Catalog-only recommendations
- Clarifying questions for vague queries
- Recommendation ranking logic
- 8-turn conversation limit
- JSON schema-compliant responses
- Swagger API documentation

## API Endpoints

### Health Check
GET `/api/health`

Response:
```json
{"status":"ok"}
```

### Chat Endpoint
POST `/api/chat`

Request:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "I need assessments for Java developers"
    }
  ]
}
```

Response:
```json
{
  "reply": "Could you clarify the seniority level and required technologies?",
  "recommendations": [],
  "end_of_conversation": false
}
```

## Data Sources

The backend uses:
- SHL Product Catalog
- C1–C10 sample conversation traces
- Assignment specification PDFs

## Tech Stack

- Python
- FastAPI
- Uvicorn

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open:
- `/api/docs`
- `/api/health`
- `/api/chat`

## Deployment

The project is deployment-ready for:
- Render
- Railway
- Replit
- Vercel (with adaptation if needed)

## Assignment Requirements Covered

- Exact JSON response schema
- Catalog-only recommendations
- No hallucinated products
- Clarifying questions
- Off-topic refusal handling
- Comparison support
- Turn limit enforcement
- Stateless API behavior

## Live API

Deployed public endpoint:
(Add deployed URL here)

## Author

Aryan Arya
