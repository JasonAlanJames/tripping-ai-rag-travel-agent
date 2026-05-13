import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus


DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "local_businesses.json"


def load_business_data() -> list[dict[str, Any]]:
    """
    Load local business/location data from backend/data/local_businesses.json.

    This is the local JSON source of truth for the starter RAG system.
    """
    if not DATA_FILE.exists():
        print(f"[RAG] Data file not found: {DATA_FILE}")
        return []

    try:
        with DATA_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            print("[RAG] Invalid data format. Expected a JSON array.")
            return []

        print(f"[RAG] Loaded {len(data)} business/location records.")
        return data

    except json.JSONDecodeError as error:
        print(f"[RAG] JSON decode error in {DATA_FILE}: {error}")
        return []

    except OSError as error:
        print(f"[RAG] File read error for {DATA_FILE}: {error}")
        return []


def normalize_text(value: str) -> str:
    """
    Normalize text for basic keyword matching.
    """
    if not value:
        return ""

    return value.lower().strip()


def tokenize_query(query: str) -> list[str]:
    """
    Convert a query string into searchable keyword tokens.

    Keeps meaningful words and removes short/noisy terms.
    """
    normalized_query = normalize_text(query)

    tokens = re.findall(r"[a-zA-Z0-9]+", normalized_query)

    stop_words = {
        "the",
        "and",
        "for",
        "from",
        "with",
        "this",
        "that",
        "into",
        "trip",
        "day",
        "days",
        "create",
        "itinerary",
        "include",
        "travel",
        "budget",
    }

    return [
        token
        for token in tokens
        if len(token) >= 3 and token not in stop_words
    ]


def address_to_google_maps_link(address: str) -> str:
    """
    Convert a human-readable address into the required Google Maps HTML link format.

    Example:
    204 W Kendall St, Corona, CA 92882

    Becomes:
    <a href="https://www.google.com/maps/place/204+W+Kendall+St+Corona+CA+92882/">204 W Kendall St, Corona, CA 92882</a>
    """
    if not address or address == "Not available":
        return "Not available"

    encoded_address = quote_plus(address.replace(",", ""))

    return (
        f'<a href="https://www.google.com/maps/place/{encoded_address}/">'
        f"{address}"
        f"</a>"
    )


def build_searchable_text(business: dict[str, Any]) -> str:
    """
    Build a searchable text blob from a business/location record.
    """
    fields = [
        business.get("name", ""),
        business.get("category", ""),
        business.get("address", ""),
        business.get("city", ""),
        business.get("state", ""),
        business.get("description", ""),
        business.get("notes", ""),
    ]

    return normalize_text(" ".join(str(field) for field in fields if field))


def business_matches_query(business: dict[str, Any], query: str) -> bool:
    """
    Basic keyword matcher for starter local JSON RAG.

    This is intentionally simple and transparent. It can later be replaced with
    Chroma, FAISS, pgvector, or another vector search backend.
    """
    query_normalized = normalize_text(query)
    searchable_text = build_searchable_text(business)
    query_tokens = tokenize_query(query)

    city = normalize_text(str(business.get("city", "")))
    state = normalize_text(str(business.get("state", "")))
    category = normalize_text(str(business.get("category", "")))
    name = normalize_text(str(business.get("name", "")))

    if city and city in query_normalized:
        return True

    if state and state in query_normalized:
        return True

    if category and category in query_normalized:
        return True

    if name and name in query_normalized:
        return True

    return any(token in searchable_text for token in query_tokens)


def format_source_record(business: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a raw business/location record into a clean structured RAG source.
    """
    address = business.get("address", "Not available")

    return {
        "name": business.get("name", "Not available"),
        "category": business.get("category", "Not available"),
        "address": address,
        "address_link": address_to_google_maps_link(address),
        "city": business.get("city", "Not available"),
        "state": business.get("state", "Not available"),
        "description": business.get("description", "Not available"),
        "notes": business.get("notes", "Not available"),
    }


def retrieve_sources(query: str, limit: int = 6) -> list[dict[str, Any]]:
    """
    Retrieve structured source records from the local business/location dataset.
    """
    print(f"[RAG] Retrieval query: {query}")

    businesses = load_business_data()

    if not businesses:
        print("[RAG] No business/location records available.")
        return []

    matches = []

    for business in businesses:
        if business_matches_query(business, query):
            matches.append(format_source_record(business))

    limited_matches = matches[:limit]

    print(f"[RAG] Matched {len(limited_matches)} source(s).")

    return limited_matches


def retrieve_context(query: str) -> str:
    """
    Retrieve RAG context as text for the LLM.

    The itinerary LLM should treat this context as the source of truth.
    """
    sources = retrieve_sources(query)

    if not sources:
        return ""

    context_blocks = []

    for index, item in enumerate(sources, start=1):
        context_blocks.append(
            f"""
SOURCE {index}
Name: {item["name"]}
Category: {item["category"]}
Address: {item["address"]}
Address Link: {item["address_link"]}
City: {item["city"]}
State: {item["state"]}
Description: {item["description"]}
Notes: {item["notes"]}
""".strip()
        )

    return "\n\n---\n\n".join(context_blocks)