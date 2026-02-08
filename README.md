# Ultra Doc-Intelligence

A quick POC for analyzing logistics documents using RAG (Retrieval Augmented Generation).

Upload a Bill of Lading, Rate Confirmation, or similar doc and ask questions about it. The system finds relevant sections and uses an LLM to answer, with confidence scoring to catch hallucinations.

## What it does

- **Upload docs** - PDF, DOCX, TXT up to 10MB
- **Ask questions** - Natural language Q&A grounded in the document
- **Extract data** - Pulls structured shipment info into JSON
- **Confidence scores** - Multi-signal scoring to flag uncertain answers
- **Guardrails** - Refuses to answer if it's not confident

## Quick start

```bash
# Backend
cd backend
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key" > .env
python -m uvicorn main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Tech stack

- **Backend**: FastAPI + Python 3.10+
- **LLM**: Google Gemini 1.5 Flash
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB (local file storage)
- **Frontend**: React + Vite

## How the RAG works

```
Document → Parse → Chunk → Embed → Store in ChromaDB
                                          ↓
Question → Embed → Search similar chunks → Build context → Gemini → Answer
```

### Chunking

Tried to be smart about it - splits on paragraph boundaries rather than arbitrary character counts. Uses 512 token chunks with 50 token overlap so we don't lose context at boundaries.

### Confidence scoring

Four signals, weighted average:
- Retrieval similarity (30%) - how close are the top chunks to the query
- Chunk agreement (25%) - do the retrieved chunks agree with each other
- Answer coverage (25%) - is the answer grounded in the context
- Q-A relevance (20%) - does the answer actually address the question

### Guardrails

Three checks to prevent hallucination:
1. **Low retrieval** - if similarity < 0.15, refuse to answer
2. **Low confidence** - if score < 0.5, add warning
3. **Poor grounding** - if answer isn't well-supported by context, flag it

## API endpoints

| Endpoint | Method | What it does |
|----------|--------|--------------|
| `/api/upload` | POST | Upload a document |
| `/api/ask` | POST | Ask a question |
| `/api/extract` | POST | Extract structured data |
| `/docs` | GET | Swagger docs |

## Structured extraction

Pulls these fields from logistics docs:
- shipment_id, shipper, consignee
- pickup/delivery dates
- equipment type, mode
- rate, currency, weight
- carrier name

Returns JSON with nulls for missing fields.

## Known limitations

- Single document at a time (no multi-doc RAG yet)
- English only
- Scanned PDFs with poor OCR might not work great
- Complex tables can chunk weirdly

## Stuff I'd add with more time

- Multi-document queries
- Better table extraction
- User feedback loop to improve accuracy
- Caching for frequently asked questions
- Rate limiting

---

Built for a coding assessment. Not production-ready but demonstrates the core concepts.
