from dotenv import load_dotenv
from app.agents.travel_agent_graph import build_graph

load_dotenv()

# Build once
graph = build_graph()


def generate_chat_response(request):
    """
    General-purpose chat endpoint for travel, food, entertainment.
    Uses the same LangGraph pipeline.
    """

    query = request.message.strip()

    result = graph.invoke({
        "query": query
    })

    return {
        "response": result.get("itinerary", "No response generated"),
        "sources_used": result.get("sources_used", 0)
    }