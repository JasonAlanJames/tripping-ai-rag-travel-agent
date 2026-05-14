# Tripping AI RAG Travel Agent

## Overview
This project is a production-grade Retrieval-Augmented Generation (RAG) travel assistant built with FastAPI, LangChain, and Google Places API. It intelligently retrieves and ranks real-world businesses (e.g., restaurants) using a hybrid pipeline combining vector search, web search, and live enrichment.

## Features
- Hybrid RAG pipeline (Vector DB + Web fallback)
- Business entity extraction and validation
- Google Places API enrichment (ratings, address, phone, website)
- Intelligent filtering and deduplication
- Ranking layer for high-quality results
- FastAPI chat endpoint
- Clean, production-ready logging
- Caching for performance optimization

## Tech Stack
- Python 3.13
- FastAPI
- LangChain
- ChromaDB (vector store)
- OpenAI Embeddings
- Tavily Search API
- Google Places API

## Project Structure
```
backend/
  app/
    agents/
    routes/
    services/
    main.py
```

## Setup

### 1. Clone Repo
```
git clone https://github.com/YOUR_USERNAME/tripping-ai-rag-travel-agent.git
cd tripping-ai-rag-travel-agent
```

### 2. Create Virtual Environment
```
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in `backend/`:

```
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
GOOGLE_PLACES_API_KEY=your_google_key
```

### 5. Run the App
```
cd backend
uvicorn app.main:app --reload
```

### 6. Test API
Open:
```
http://127.0.0.1:8000/docs
```

Use `/chat` endpoint.

## Example Query
```
What are the best sushi restaurants in Corona, CA?
```

## Current Capabilities
- Returns real businesses with ratings and contact info
- Filters out junk and non-business entities
- Uses live Google data for accuracy
- Provides ranked results

## Roadmap
- Distance-based ranking
- “Open now” filtering
- Review count weighting
- Multi-source confidence scoring
- Frontend UI

## Author
Jason A. James, B.S. Computer Information Technology

## License
MIT
