from dotenv import load_dotenv
from app.agents.travel_agent_graph import build_graph
from app.services.rag_service import retrieve_context

load_dotenv()

# Build once
graph = build_graph()


def generate_chat_response(request):
    """
    General-purpose chat endpoint for travel, food, entertainment.
    Now properly integrates RAG pipeline before invoking graph.
    """

    query = request.message.strip()

    # -----------------------------------------------------
    # 🔥 STEP 1: Run RAG retrieval (CRITICAL FIX)
    # -----------------------------------------------------
    context, confidence = retrieve_context(query)

    print(f"[CHAT] Retrieved context (confidence={confidence})")

    # -----------------------------------------------------
    # 🔥 STEP 2: Pass context into graph
    # -----------------------------------------------------
    try:
        result = graph.invoke({
            "query": query,
            "context": context,   # <-- KEY FIX
            "confidence": confidence
        })
    except Exception as e:
        print(f"[CHAT] Graph error: {e}")
        result = {}

    # -----------------------------------------------------
    # 🔥 STEP 3: Response selection logic
    # -----------------------------------------------------
    response = result.get("itinerary")

    # Fallback to RAG context if graph fails or is generic
    if not response or len(response.strip()) < 50:
        print("[CHAT] Using RAG fallback response")
        response = context

    return {
        "response": response,
        "sources_used": result.get("sources_used", 0),
        "confidence": confidence
    }