# LLM-Powered Intelligent Query-Retrieval System

## Overview
An advanced document processing and query retrieval system that handles insurance, legal, HR, and compliance documents using semantic search and LLM-powered decision making.

## Features
- Multi-format document processing (PDF, DOCX, Email)
- Semantic search using FAISS/Pinecone embeddings
- Clause matching and retrieval
- Explainable AI with decision rationale
- RESTful API with structured JSON responses
- Token-efficient LLM integration

## Tech Stack
- **Backend**: FastAPI
- **Vector Database**: Pinecone (with FAISS fallback)
- **LLM**: GPT-4
- **Database**: PostgreSQL
- **Document Processing**: PyPDF2, python-docx
- **Embeddings**: OpenAI Ada-002

## System Architecture
```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Input Documents │────►│ LLM Parser   │────►│ Embedding Search│
│ (PDF/DOCX/Email)│     │ (GPT-4)      │     │ (Pinecone/FAISS)│
└─────────────────┘     └──────────────┘     └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  JSON Output    │◄────│Logic Eval    │◄────│ Clause Matching │
│  (Structured)   │     │(Decision)    │     │ (Semantic Sim)  │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-query-retrieval-system.git
cd llm-query-retrieval-system
```

2. Create virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. Initialize database:
```bash
python scripts/init_db.py
```

6. Run the application:
```bash
uvicorn app.main:app --reload --port 8000
```

## API Usage

### Submit Query
```bash
POST /api/v1/hackrx/run
Content-Type: application/json
Authorization: Bearer YOUR_TOKEN

{
    "documents": "https://example.com/policy.pdf",
    "questions": [
        "What is the coverage for knee surgery?",
        "What are the policy conditions?"
    ]
}
```

### Response Format
```json
{
    "answers": [
        "The policy covers knee surgery with...",
        "The policy conditions include..."
    ],
    "metadata": {
        "processing_time": 2.3,
        "tokens_used": 1500,
        "confidence_scores": [0.95, 0.87]
    }
}
```

## Project Structure
```
llm-query-retrieval-system/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   └── dependencies.py
│   ├── core/
│   │   ├── config.py
│   │   └── security.py
│   ├── models/
│   ├── services/
│   │   ├── document_processor.py
│   │   ├── embedding_service.py
│   │   ├── llm_service.py
│   │   └── query_engine.py
│   └── main.py
├── tests/
├── scripts/
├── requirements.txt
└── README.md
```
