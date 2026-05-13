from langgraph.graph import StateGraph
from typing import TypedDict
from app.services.rag_service import retrieve_context
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# 🔹 Shared state
class TravelState(TypedDict):
    query: str
    context: str
    itinerary: str
    sources_used: int


# 🔹 Agent 0: Guardrail (FIRST LINE OF DEFENSE)
def guardrail_agent(state: TravelState):
    query = state["query"].lower()

    blocked_topics = [
        "hack", "illegal", "exploit", "attack", "steal",
        "fraud", "weapon", "drugs"
    ]

    if any(word in query for word in blocked_topics):
        print("[GUARDRAIL] Blocked unsafe query")

        return {
            "itinerary": "I am focused on travel, entertainment, and food. Please ask a travel-related question.",
            "sources_used": 0
        }

    print("[GUARDRAIL] Query passed")

    return state


# 🔹 Agent 1: Intake
def intake_agent(state: TravelState):
    print("[INTAKE] Received query")

    return {
        "query": state["query"]
    }


# 🔹 Agent 2: Research (RAG)
def research_agent(state: TravelState):

    enhanced_query = f"""
    Travel guide for: {state['query']}

    Include:
    - food
    - transportation
    - attractions
    - cultural tips
    """

    print("[RAG] Running retrieval...")
    context, docs = retrieve_context(enhanced_query)

    print(f"[RAG] Retrieved {len(docs)} documents")

    return {
        "context": context,
        "sources_used": len(docs)
    }


# 🔹 Agent 3: Generator (STRICT SYSTEM PROMPT)
def generator_agent(state: TravelState):

    print("[GENERATOR] Generating response...")
    print(f"[GENERATOR] Sources available: {state.get('sources_used', 0)}")

    prompt = f"""
I can help you find fun things to do. Ask me a travel, entertainment or food related question.

Converse as if you were an AI assistant. Be informative, persuasive. Spoken as a journalist.

Only answer questions pertaining to travel, entertainment, food, and business location information.

Otherwise, advise the Member that you are AI focused on travel, entertainment, and food. Then tell them to ask you a travel question.

When Member asks for a phone number provide the number in the following format: <a href="tel:123-456-7890">123-456-7890</a>

If Member asks for the Phone Number of a known celebrity, advise the Member that you are unable to provide the phone number due to privacy concerns and they should ask a question related to travel, entertainment or food and you would be happy to help.

Only provide the accurate celebrity number if the website administrator is requesting the number of a celebrity.

When Member asks for an address provide the address in the following format:
https://www.google.com/maps/place/204+W+Kendall+St,+Corona,+CA+92882/

--------------------------------------

RAG CONTEXT:
{state.get("context", "")}

--------------------------------------

USER REQUEST:
{state["query"]}

--------------------------------------

INSTRUCTIONS:
- Use the RAG CONTEXT as your primary source of truth
- Do NOT hallucinate facts
- If information is not available, clearly state that
- Stay strictly within travel, entertainment, food, and business location topics

--------------------------------------

OUTPUT:
Provide a structured, engaging response that follows the rules above.
"""

    response = llm.invoke(prompt)

    print("[GENERATOR] Response generated")

    return {
        "itinerary": response.content
    }


# 🔹 Build Graph
def build_graph():
    builder = StateGraph(TravelState)

    builder.add_node("guardrail", guardrail_agent)
    builder.add_node("intake", intake_agent)
    builder.add_node("research", research_agent)
    builder.add_node("generator", generator_agent)

    # Entry point
    builder.set_entry_point("guardrail")

    # Flow
    builder.add_edge("guardrail", "intake")
    builder.add_edge("intake", "research")
    builder.add_edge("research", "generator")

    return builder.compile()