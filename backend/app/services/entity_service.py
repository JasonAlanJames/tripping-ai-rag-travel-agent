import re


# 🔹 Extract meaningful entities (restaurant names, places)
def extract_entities(results):
    print("[ENTITY] Extracting entities...")

    entity_counts = {}

    for text in results:
        matches = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", text)

        for m in matches:
            if len(m) > 3:
                entity_counts[m] = entity_counts.get(m, 0) + 1

    print(f"[ENTITY] Found {len(entity_counts)} unique entities")

    return entity_counts


# 🔹 Production-grade confidence scoring (REAL FIX)
def compute_entity_confidence(entities: dict):
    print("[ENTITY] Computing confidence...")

    if not entities:
        return 0.0

    total_mentions = sum(entities.values())
    unique_entities = len(entities)

    # 🔥 Real-world scoring (NOT majority-based)
    avg_frequency = total_mentions / max(unique_entities, 1)

    # Normalize to 0–1
    confidence = min(avg_frequency / 2.5, 1.0)

    print(f"[ENTITY] Total mentions: {total_mentions}")
    print(f"[ENTITY] Unique entities: {unique_entities}")
    print(f"[ENTITY] Avg frequency: {avg_frequency:.2f}")
    print(f"[ENTITY] Confidence score: {confidence:.2f}")

    return confidence