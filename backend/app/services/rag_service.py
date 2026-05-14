# =========================================================
# IMPORTS
# =========================================================

import re
import time
from typing import List, Any, Tuple

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.services.search_service import tavily_search
from app.services.google_places_service import enrich_businesses


# =========================================================
# VECTOR STORE
# =========================================================

embedding = OpenAIEmbeddings()

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

def detect_intent(query: str) -> str:
    q = query.lower()

    if any(x in q for x in ["things to do", "activities", "fun", "kids", "family"]):
        return "activities"

    if any(x in q for x in ["restaurants", "food", "eat", "dining"]):
        return "restaurants"

    return "general"


# =========================================================
# CACHE
# =========================================================

RAG_CACHE = {}
CACHE_TTL_SECONDS = 3600


def is_cache_valid(entry):
    return time.time() - entry["timestamp"] < CACHE_TTL_SECONDS


# =========================================================
# UTILITIES
# =========================================================

def extract_location(query: str) -> str:
    """
    Extracts a clean location from a natural language query.
    Prevents over-capturing trailing phrases.
    Example:
    "things to do in San Diego with kids" → "san diego"
    """

    query = query.lower()

    # Non-greedy match, stops at punctuation or sentence boundary
    match = re.search(r"in ([a-z\s]+?)(?:\?|,|\.|$)", query)

    if match:
        location = match.group(1).strip()

        # Safety cleanup for common trailing noise words
        stop_words = [
            "that", "with", "for", "and", "near", "around"
        ]

        tokens = location.split()

        # Trim trailing stop words
        while tokens and tokens[-1] in stop_words:
            tokens.pop()

        return " ".join(tokens)

    return ""


def trim_context(context: str, max_chars: int = 3000) -> str:
    return context[:max_chars]


# =========================================================
# BUSINESS VALIDATOR
# =========================================================

def is_valid_business_candidate(name: str) -> bool:
    if not name:
        return False

    lower = name.lower().strip()

    blocked = [
        "county", "city", "california", "about", "share", "link",
        "menu", "gallery", "directions", "visit", "experience",
        "best", "top", "cheap"
    ]

    if any(b in lower for b in blocked):
        return False

    generic_exact = [
        "japanese restaurant",
        "chinese restaurant",
        "mexican restaurant",
        "sushi restaurant",
        "japanese izakaya",
        "sushi bar",
        "sushi bars",
        "restaurant",
        "restaurants"
    ]

    if lower in generic_exact:
        return False

    generic_phrases = [
        "food", "places", "locations",
        "all you can eat"
    ]

    if any(g in lower for g in generic_phrases):
        return False

    if "'s" in lower or "yelp" in lower or "linkedin" in lower:
        return False

    words = name.split()

    if len(words) < 2 or len(words) > 5:
        return False

    strong_tokens = ["sushi", "kitchen", "grill", "cafe", "ramen"]

    if not any(token in lower for token in strong_tokens):
        return False

    return True


# =========================================================
# LOCATION VALIDATOR
# =========================================================

def is_location_relevant(name: str, location: str) -> bool:
    if not location:
        return True

    lower = name.lower()

    if location not in lower:
        other_cities = [
            "whittier", "long beach", "los angeles",
            "anaheim", "irvine", "san diego"
        ]

        if any(city in lower for city in other_cities):
            return False

    return True


# =========================================================
# RANKING
# =========================================================

def rank_businesses(businesses: List[dict]) -> List[dict]:
    def score(b):
        rating = b.get("rating", 0) or 0
        name = b.get("name", "").lower()
        address = b.get("address", "").lower()

        score = rating

        if "sushi" in name:
            score += 1.5

        if "corona" in address:
            score += 2

        if name in ["sushi bar", "sushi bars"]:
            score -= 2

        if not b.get("address"):
            score -= 1
        if not b.get("phone"):
            score -= 0.5

        return score

    ranked = sorted(businesses, key=score, reverse=True)

    seen = set()
    deduped = []

    for b in ranked:
        name = b.get("name", "").lower()
        normalized = re.sub(r'\b(restaurant|sushi|bar)\b', '', name).strip()

        if normalized not in seen:
            seen.add(normalized)
            deduped.append(b)

    return deduped


# =========================================================
# ENSURE MINIMUM CANDIDATES
# =========================================================

def ensure_minimum_candidates(candidates: List[str], query: str, location: str) -> List[str]:
    if len(candidates) >= 5:
        return candidates

    print("[RAG] Expanding candidates for coverage")

    fallback_terms = [
        f"best restaurants in {location}",
        f"top rated restaurants in {location}",
        query
    ]

    expanded = set(candidates)

    for term in fallback_terms:
        results = tavily_search(term)
        extra = extract_candidate_names(results)

        for name in extra:
            if is_valid_business_candidate(name) and is_location_relevant(name, location):
                expanded.add(name)

        if len(expanded) >= 8:
            break

    return list(expanded)


# =========================================================
# KNOWLEDGE BASE
# =========================================================

def is_duplicate(context: str) -> bool:
    try:
        docs = vectorstore.similarity_search(context, k=1)
        if docs and context[:200] in docs[0].page_content:
            return True
    except Exception:
        pass
    return False


def save_to_knowledge_base(query: str, context: str, confidence: float):
    if confidence < 0.6 or is_duplicate(context):
        return

    try:
        vectorstore.add_texts([context], metadatas=[{"query": query}])
        print("[RAG] Knowledge saved")
    except Exception as e:
        print(f"[RAG] Save error: {e}")


# =========================================================
# PROCESS WEB RESULTS
# =========================================================

def process_web_results(results: List[Any], query: str, location: str):
    texts = []

    for r in results:
        text = r.get("content") if isinstance(r, dict) else str(r)
        texts.append(text)

    context = "\n\n".join(texts)

    if not context.strip():
        return "", 0.0

    return trim_context(context), 1.0


# =========================================================
# NAME EXTRACTION
# =========================================================

def extract_candidate_names(results: List[Any]) -> List[str]:
    candidates = set()

    pattern = r'\b([A-Z][a-zA-Z&\'\-]+(?:\s+[A-Z][a-zA-Z&\'\-]+){1,4})\b'

    for r in results:
        text = str(r)
        matches = re.findall(pattern, text)

        for name in matches:
            candidates.add(name.strip())

    return list(candidates)


# =========================================================
# MAIN PIPELINE
# =========================================================

def retrieve_context(query: str) -> Tuple[str, float]:
    print("[RAG] Running retrieval...")

    intent = detect_intent(query)
    print(f"[RAG] Intent detected: {intent}")

    if intent == "activities":
        print("[RAG] Activity query → skipping restaurant pipeline")

    results = tavily_search(query)

    context = "\n\n".join([
        r.get("content", "") for r in results if r.get("content")
    ])

    confidence = 1.0  # ✅ CRITICAL FIX

    return context, confidence

    location = extract_location(query)
    cache_key = f"{query.lower()}::{location.lower()}"

    print(f"[RAG] Query: {query}")
    print(f"[RAG] Location: {location}")

    if cache_key in RAG_CACHE:
        entry = RAG_CACHE[cache_key]
        if is_cache_valid(entry):
            print("[RAG] Using cached result")
            return entry["context"], entry["confidence"]
        del RAG_CACHE[cache_key]

    try:
        docs = vectorstore.similarity_search(query, k=3)
    except Exception:
        docs = []

    if docs:
        combined = "\n\n".join([d.page_content for d in docs])

        if location and location in combined.lower():
            print("[RAG] Using vectorstore (with enrichment)")

            candidates = extract_candidate_names([combined])
            print(f"[RAG] Raw candidates: {candidates}")

            filtered_candidates = ensure_minimum_candidates(
                [
                    c for c in candidates
                    if is_valid_business_candidate(c)
                    and is_location_relevant(c, location)
                ],
                query,
                location
            )

            print(f"[RAG] Filtered candidates: {filtered_candidates}")
            print(f"[GOOGLE] Enriching: {filtered_candidates[:5]}")

            enriched = enrich_businesses(filtered_candidates[:5], location)

            if enriched:
                enriched = rank_businesses(enriched)

                lines = ["\nTop verified restaurants:\n"]

                for b in enriched:
                    line = f"- {b['name']}"
                    if b.get("rating"):
                        line += f" (⭐ {b['rating']})"
                    if b.get("address"):
                        line += f"\n  Address: {b['address']}"
                    if b.get("phone"):
                        line += f"\n  Phone: {b['phone']}"
                    lines.append(line)

                context = "\n".join(lines)
                confidence = 0.9
            else:
                context = trim_context(combined)
                confidence = 0.85

            RAG_CACHE[cache_key] = {
                "context": context,
                "confidence": confidence,
                "timestamp": time.time()
            }

            return context, confidence

    print("[RAG] Running Tavily pipeline")

    from app.agents.travel_agent_graph import expand_query

    all_results = []

    for q in expand_query(query):
        all_results.extend(tavily_search(q))

    unique_results = list({str(r): r for r in all_results}.values())

    candidates = extract_candidate_names(unique_results)
    print(f"[RAG] Raw candidates: {candidates}")

    filtered_candidates = ensure_minimum_candidates(
        [
            c for c in candidates
            if is_valid_business_candidate(c)
            and is_location_relevant(c, location)
        ],
        query,
        location
    )

    print(f"[RAG] Filtered candidates: {filtered_candidates}")
    print(f"[GOOGLE] Enriching: {filtered_candidates[:5]}")

    enriched = enrich_businesses(filtered_candidates[:5], location)

    if enriched:
        enriched = rank_businesses(enriched)

        lines = ["\nTop verified restaurants:\n"]

        for b in enriched:
            line = f"- {b['name']}"
            if b.get("rating"):
                line += f" (⭐ {b['rating']})"
            if b.get("address"):
                line += f"\n  Address: {b['address']}"
            if b.get("phone"):
                line += f"\n  Phone: {b['phone']}"
            lines.append(line)

        context = "\n".join(lines)
        confidence = 1.0
    else:
        context, confidence = process_web_results(unique_results, query, location)

    save_to_knowledge_base(query, context, confidence)

    RAG_CACHE[cache_key] = {
        "context": context,
        "confidence": confidence,
        "timestamp": time.time()
    }

    return context, confidence


# =========================================================
# SOURCES
# =========================================================

def retrieve_sources(query: str):
    from app.agents.travel_agent_graph import expand_query

    all_results = []

    for q in expand_query(query):
        all_results.extend(tavily_search(q))

    return all_results[:5]