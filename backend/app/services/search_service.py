import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found")

tavily = TavilyClient(api_key=TAVILY_API_KEY)


def tavily_search(query: str, max_results: int = 5):
    """
    Perform Tavily search with:
    - deduplication
    - content filtering
    - structured output for verification engine
    """

    print("[SEARCH] Running Tavily search...")
    print(f"[SEARCH] Query: {query}")

    try:
        response = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=max_results
        )

        raw_results = response.get("results", [])

        if not raw_results:
            print("[SEARCH] No results found")
            return []

        seen_urls = set()
        cleaned_results = []

        for r in raw_results:
            title = r.get("title", "")
            content = r.get("content") or r.get("snippet") or ""
            url = r.get("url", "")

            # 🔹 Skip duplicates
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            # 🔹 Skip weak content
            if not content or len(content) < 50:
                continue

            cleaned_results.append({
                "title": title.strip(),
                "content": content.strip(),
                "url": url.strip()
            })

        print(f"[SEARCH] Retrieved {len(cleaned_results)} clean results")

        return cleaned_results

    except Exception as e:
        print(f"[SEARCH ERROR] {e}")
        return []