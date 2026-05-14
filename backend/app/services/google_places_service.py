import os
import requests
from typing import List, Dict, Optional
from difflib import SequenceMatcher

# ----------------------------
# CONFIG
# ----------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

if not GOOGLE_API_KEY:
    print("[GOOGLE] WARNING: GOOGLE_PLACES_API_KEY not set")

BASE_URL = "https://maps.googleapis.com/maps/api/place"
MATCH_THRESHOLD = 0.70
MAX_RESULTS = 5


# ----------------------------
# SIMILARITY
# ----------------------------
def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ----------------------------
# NAME VALIDATION (RELAXED + SAFE)
# ----------------------------
def is_valid_business_name(name: str) -> bool:
    if not name:
        return False

    lower = name.lower().strip()

    # 🚫 PLATFORM / NAVIGATION
    noise = [
        "yelp", "tripadvisor", "doordash",
        "menu", "order", "delivery", "pickup",
        "hours", "locations"
    ]

    if any(n in lower for n in noise):
        return False

    # 🚫 MARKETING / JUNK
    junk = [
        "best", "top", "near me"
    ]

    if any(j in lower for j in junk):
        return False

    # STRUCTURE
    words = name.split()
    if len(words) < 2 or len(words) > 6:
        return False

    return True


# ----------------------------
# BEST MATCH (IMPROVED)
# ----------------------------
def find_best_match(name: str, results: List[Dict], location: str) -> Optional[Dict]:
    best = None
    best_score = 0.0

    for r in results:
        candidate_name = r.get("name", "")
        candidate_address = r.get("formatted_address", "")

        if not candidate_name:
            continue

        name_score = similarity(name, candidate_name)

        address_lower = candidate_address.lower()
        location_lower = location.lower()

        if location_lower in address_lower:
            location_score = 1.0
        else:
            if any(city in address_lower for city in ["long beach", "los angeles", "anaheim", "san diego"]):
                return None
            location_score = 0.2

        total_score = (name_score * 0.8) + (location_score * 0.2)

        if total_score > best_score:
            best = r
            best_score = total_score

    if best_score < MATCH_THRESHOLD:
        return None

    return best


# ----------------------------
# SEARCH (WITH RETRY)
# ----------------------------
def search_places(query: str, location: str) -> List[Dict]:
    if not GOOGLE_API_KEY:
        return []

    print(f"[GOOGLE] Searching: {query}")

    try:
        params = {
            "query": f"{query} in {location}",
            "key": GOOGLE_API_KEY
        }

        response = requests.get(
            f"{BASE_URL}/textsearch/json",
            params=params,
            timeout=6
        )

        if response.status_code != 200:
            print("[GOOGLE] HTTP Error:", response.status_code)
            return []

        data = response.json()

        if data.get("status") not in ["OK", "ZERO_RESULTS"]:
            print("[GOOGLE] API Status:", data.get("status"))
            return []

        return data.get("results", [])

    except Exception as e:
        print("[GOOGLE] Search Exception:", str(e))
        return []


# ----------------------------
# DETAILS
# ----------------------------
def get_place_details(place_id: str) -> Optional[Dict]:
    if not GOOGLE_API_KEY:
        return None

    try:
        params = {
            "place_id": place_id,
            "fields": "name,formatted_address,rating,types,website,formatted_phone_number",
            "key": GOOGLE_API_KEY
        }

        response = requests.get(
            f"{BASE_URL}/details/json",
            params=params,
            timeout=6
        )

        if response.status_code != 200:
            return None

        data = response.json()

        if data.get("status") != "OK":
            return None

        return data.get("result")

    except Exception as e:
        print("[GOOGLE] Details Exception:", str(e))
        return None


# ----------------------------
# MAIN PIPELINE (PRODUCTION)
# ----------------------------
def enrich_businesses(business_names: List[str], location: str) -> List[Dict]:

    enriched = []
    seen_ids = set()

    # ----------------------------
    # CLEAN INPUT
    # ----------------------------
    cleaned = [b for b in business_names if is_valid_business_name(b)]

    if not cleaned:
        print("[GOOGLE] No valid inputs")
        return []

    # ----------------------------
    # QUERY STRATEGY (EXPANDED)
    # ----------------------------
    queries = []

    for name in cleaned:
        queries.append(name)
        queries.append(f"{name} {location}")

    queries.append(f"restaurants in {location}")

    # ----------------------------
    # EXECUTION
    # ----------------------------
    for q in queries:
        results = search_places(q, location)

        if not results:
            continue

        for r in results:
            place_id = r.get("place_id")

            if not place_id or place_id in seen_ids:
                continue

            best = find_best_match(q, [r], location)

            if not best:
                continue

            seen_ids.add(place_id)

            details = get_place_details(place_id)

            if not details:
                continue

            final_name = details.get("name")

            if not is_valid_business_name(final_name):
                continue

            enriched.append({
                "name": final_name,
                "address": details.get("formatted_address"),
                "rating": details.get("rating"),
                "phone": details.get("formatted_phone_number"),
                "website": details.get("website"),
                "types": details.get("types", [])
            })

            if len(enriched) >= MAX_RESULTS:
                break

        if len(enriched) >= MAX_RESULTS:
            break

    # ----------------------------
    # FINAL SAFETY
    # ----------------------------
    if len(enriched) == 0:
        print("[GOOGLE] ❌ No results")

    elif len(enriched) < 3:
        print("[GOOGLE] ⚠️ Low result count:", len(enriched))

    return enriched