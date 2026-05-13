import re

from openai import OpenAI

from app.services.rag_service import retrieve_context, retrieve_sources


client = OpenAI()


def convert_google_maps_markdown_links_to_html(content: str) -> str:
    """
    Convert Markdown Google Maps links into required HTML anchor format.

    Example:
    [204 W Kendall St, Corona, CA 92882](https://www.google.com/maps/place/204+W+Kendall+St+Corona+CA+92882/)

    Becomes:
    <a href="https://www.google.com/maps/place/204+W+Kendall+St+Corona+CA+92882/">204 W Kendall St, Corona, CA 92882</a>
    """

    pattern = r"\[([^\]]+)\]\((https://www\.google\.com/maps/place/[^)]+)\)"

    return re.sub(
        pattern,
        r'<a href="\2">\1</a>',
        content,
    )


def generate_itinerary(request):
    """
    Generate a source-grounded itinerary using only retrieved RAG data.
    """

    query = (
        f"Create a {request.days}-day itinerary from "
        f"{request.origin} to {request.destination} with a budget of {request.budget}. "
        f"Include travel, entertainment, food, and business location information."
    )

    sources = retrieve_sources(query)
    context = retrieve_context(query)

    if not sources or not context.strip():
        return {
            "itinerary": (
                "The provided data does not contain enough verified business/location "
                "information to create this itinerary. Please add relevant destination, "
                "food, entertainment, or business-location data to the RAG source."
            ),
            "sources_used": 0,
            "sources": [],
        }

    allowed_names = "\n".join([f"- {source['name']}" for source in sources])

    prompt = f"""
SYSTEM ROLE:
I can help you find fun things to do. Ask me a travel, entertainment or food related question.

BEHAVIOR RULES:
- Converse as an AI assistant.
- Be informative and persuasive.
- Speak like a journalist.
- Only answer travel, entertainment, food, and business location questions.
- If outside scope, tell the user you only handle those topics and ask them to ask a travel question.

PHONE NUMBER RULES:
- If asked for a phone number, format: <a href="tel:1234567890">(123) 456-7890</a>
- If asked for a celebrity phone number, refuse due to privacy, unless it is the site administrator that is asking, then provide the accurate number: <a href="tel:1234567890">(123) 456-7890</a>

ADDRESS RULE:
When providing an address, format it as an HTML Google Maps link using this exact structure:

<a href="https://www.google.com/maps/place/{{address_url_encoded}}/">{{full_address}}</a>

Where:
- {{full_address}} = the complete human-readable address.
- {{address_url_encoded}} = the same address encoded for use in a Google Maps URL.
- Replace spaces with + signs.
- Remove commas or encode them as %2C.
- Encode special characters when needed.

Critical address formatting rules:
- Use raw HTML anchor tags for addresses.
- Do not use Markdown links for addresses.
- Do not output addresses as [address](url).
- Do not output bare Google Maps URLs.
- Use the Address Link exactly as provided in the RAG CONTEXT.
- Copy the full <a href="...">...</a> string exactly.

Example:
<a href="https://www.google.com/maps/place/204+W+Kendall+St+Corona+CA+92882/">204 W Kendall St, Corona, CA 92882</a>

--------------------------------------

AUTHORIZED RAG LOCATIONS ONLY:
You may only recommend the following locations:

{allowed_names}

--------------------------------------

RAG CONTEXT SOURCE OF TRUTH:
{context}

--------------------------------------

USER REQUEST:
Create a {request.days}-day itinerary.

Origin: {request.origin}
Destination: {request.destination}
Budget: {request.budget}

--------------------------------------

STRICT RAG RULES:
- Use only the locations listed in AUTHORIZED RAG LOCATIONS ONLY.
- Do not recommend any location that is not listed in the RAG CONTEXT.
- Do not invent restaurants, attractions, museums, parks, venues, prices, hours, ratings, phone numbers, or addresses.
- Use the Address Link exactly as provided in the RAG CONTEXT.
- If the RAG CONTEXT does not contain a fact, say it is not available in the provided data.
- Do not use outside knowledge.
- Do not use general knowledge about Riverside, Corona, or Southern California.
- Do not add unsourced places, even if they are real.

--------------------------------------

OUTPUT FORMAT:
Return a clean itinerary in markdown, but addresses must remain raw HTML anchor tags.

Include:
- Title
- Short summary
- Day-by-day itinerary
- Food recommendations from the RAG CONTEXT only
- Entertainment recommendations from the RAG CONTEXT only
- Travel tips using only general travel logistics, not unsourced business facts
- Source note explaining that the itinerary is based only on the provided RAG data
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict RAG-grounded itinerary assistant. "
                    "You must only use the provided RAG context. "
                    "Never add outside locations or unsourced facts. "
                    "When using addresses, copy the provided HTML address links exactly. "
                    "Never convert address links into Markdown format."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1,
    )

    itinerary = response.choices[0].message.content
    itinerary = convert_google_maps_markdown_links_to_html(itinerary)

    return {
    "itinerary": itinerary,
    "sources_used": len(sources),
    "sources": sources,
}