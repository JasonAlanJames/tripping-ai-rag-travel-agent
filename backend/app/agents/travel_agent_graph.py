from langgraph.graph import StateGraph
from typing import TypedDict
from app.services.rag_service import (
    retrieve_context,
    save_to_knowledge_base,
    process_web_results
)
from app.services.search_service import tavily_search
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# ----------------------------
# STATE
# ----------------------------
class TravelState(TypedDict, total=False):
    query: str
    context: str
    itinerary: str
    sources_used: int
    confidence: float
    blocked: bool


# ----------------------------
# LOCATION DETECTION
# ----------------------------
def detect_location(query: str):
    import re
    match = re.search(r"in ([a-zA-Z\s,]+)", query.lower())
    return match.group(1).strip() if match else None


# ----------------------------
# QUERY EXPANSION (COST CONTROLLED)
# ----------------------------
def expand_query(query: str, location: str = None):
    queries = [query]

    if location:
        queries.append(f"best restaurants in {location}")

    return queries[:2]


# ----------------------------
# GUARDRAIL
# ----------------------------
def guardrail_agent(state: TravelState):
    query = state.get("query", "").lower()

    blocked_topics = [
        "hack", "illegal", "exploit", "attack", "steal",
        "fraud", "weapon", "drugs"
    ]

    if any(word in query for word in blocked_topics):
        print("[GUARDRAIL] Blocked unsafe query")

        return {
            **state,
            "context": "",
            "itinerary": "I am focused on travel, entertainment, and food.",
            "sources_used": 0,
            "confidence": 0.0,
            "blocked": True
        }

    print("[GUARDRAIL] Query passed")
    return {**state, "blocked": False}


# ----------------------------
# INTAKE
# ----------------------------
def intake_agent(state: TravelState):
    if state.get("blocked"):
        return state

    print("[INTAKE] Received query")
    return state


# ----------------------------
# RESEARCH (CORE ENGINE)
# ----------------------------
def research_agent(state: TravelState):
    if state.get("blocked"):
        return state

    query = state.get("query", "")

    print("[RAG] Running retrieval...")
    print(f"[RAG] Query: {query}")

    requested_location = detect_location(query)
    print(f"[RAG] Requested location: {requested_location}")

    # ----------------------------
    # STEP 1: MEMORY CHECK
    # ----------------------------
    try:
        context, confidence = retrieve_context(query)
    except Exception as e:
        print(f"[RAG] Retrieval failed: {e}")
        context, confidence = "", 0.0

    print(f"[RAG] Context length: {len(context)}")
    print(f"[RAG] Confidence: {confidence}")

    # ----------------------------
    # STEP 2: USE CACHE IF STRONG
    # ----------------------------
    if context and confidence >= 0.6:
        print("[RAG] Using cached knowledge (NO Tavily)")
        return {
            **state,
            "context": context,
            "sources_used": int(confidence * 10),
            "confidence": confidence
        }

    # ----------------------------
    # STEP 3: TAVILY SEARCH
    # ----------------------------
    print("[RAG] Memory NOT relevant — using Tavily")

    expanded_queries = expand_query(query, requested_location)
    print(f"[SEARCH] Expanded queries: {expanded_queries}")

    all_results = []

    for q in expanded_queries:
        try:
            res = tavily_search(q)
            all_results.extend(res)
        except Exception as e:
            print(f"[SEARCH ERROR] {q}: {e}")

    # ----------------------------
    # STEP 4: NORMALIZE
    # ----------------------------
    normalized = []

    for r in all_results:
        if isinstance(r, str):
            normalized.append(r)
        elif isinstance(r, dict):
            content = r.get("content") or r.get("snippet") or r.get("title")
            if content:
                normalized.append(content)

    # ----------------------------
    # STEP 5: DEDUPLICATE
    # ----------------------------
    seen = set()
    unique = []

    for r in normalized:
        key = hash(r.strip().lower())
        if key not in seen:
            seen.add(key)
            unique.append(r)

    all_results = unique

    print(f"[SEARCH] Total merged results: {len(all_results)}")

    if not all_results:
        return {
            **state,
            "context": "",
            "sources_used": 0,
            "confidence": 0.0
        }

    # ----------------------------
    # STEP 6: RANK + FILTER + VERIFY
    # ----------------------------
    context, confidence = process_web_results(all_results, query)

    print(f"[VERIFY] Final confidence: {confidence:.2f}")

    # ----------------------------
    # STEP 7: SAVE KNOWLEDGE
    # ----------------------------
    if confidence >= 0.5:
        try:
            save_to_knowledge_base(query, context, confidence)
        except Exception as e:
            print(f"[MEMORY ERROR] {e}")
    elif confidence < 0.1:
        print("[VERIFY] Very weak confidence — not saving")
    else:
        print("[VERIFY] Medium confidence — not saving")

    return {
        **state,
        "context": context,
        "sources_used": int(confidence * 10),
        "confidence": confidence
    }


# ----------------------------
# GENERATOR
# ----------------------------
def generator_agent(state: TravelState):

    if state.get("blocked"):
        return state

    context = state.get("context", "")
    confidence = state.get("confidence", 0)

    if not context:
        return {
            **state,
            "itinerary": "No verified travel data available.",
            "sources_used": 0
        }

    response = llm.invoke(f"""
You are a travel assistant.

Confidence Score: {confidence}

Context:
{context}

User:
{state.get("query")}

Provide a clean, structured answer.
Only include businesses that clearly match the requested location.
""")

    return {
        **state,
        "itinerary": response.content
    }


# ----------------------------
# GRAPH
# ----------------------------
def build_graph():
    builder = StateGraph(TravelState)

    builder.add_node("guardrail", guardrail_agent)
    builder.add_node("intake", intake_agent)
    builder.add_node("research", research_agent)
    builder.add_node("generator", generator_agent)

    builder.set_entry_point("guardrail")

    builder.add_edge("guardrail", "intake")
    builder.add_edge("intake", "research")
    builder.add_edge("research", "generator")

    return builder.compile()