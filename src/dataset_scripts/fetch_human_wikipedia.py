import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.parse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import SCIENTIFIC_SEARCH_TERMS, DATA_RAW_GENERATED, truncate_text

OUTPUT_PATH = os.path.join(DATA_RAW_GENERATED, "human.csv")
HEADERS = {"User-Agent": "StyleDetectionResearch/1.0 (academic project)"}


def fetch_wikipedia_paragraphs(max_samples=180):
    paragraphs = []
    seen_titles = set()

    for i, topic in enumerate(SCIENTIFIC_SEARCH_TERMS):
        if len(paragraphs) >= max_samples:
            break

        print(f"  [{i+1}/{len(SCIENTIFIC_SEARCH_TERMS)}] Searching: {topic}...")

        search_url = (
            "https://en.wikipedia.org/w/api.php?"
            + urllib.parse.urlencode({
                "action": "query",
                "list": "search",
                "srsearch": topic,
                "srlimit": "5",
                "format": "json",
            })
        )

        for attempt in range(3):
            try:
                req = urllib.request.Request(search_url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())

                for result in data.get("query", {}).get("search", []):
                    if len(paragraphs) >= max_samples:
                        break

                    title = result["title"]
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)

                    extract_url = (
                        "https://en.wikipedia.org/w/api.php?"
                        + urllib.parse.urlencode({
                            "action": "query",
                            "titles": title,
                            "prop": "extracts",
                            "exintro": "true",
                            "explaintext": "true",
                            "exsectionformat": "plain",
                            "format": "json",
                        })
                    )

                    req2 = urllib.request.Request(extract_url, headers=HEADERS)
                    with urllib.request.urlopen(req2, timeout=10) as resp2:
                        page_data = json.loads(resp2.read().decode())

                    pages = page_data.get("query", {}).get("pages", {})
                    for page in pages.values():
                        extract = page.get("extract", "")
                        words = extract.split()
                        if len(words) >= 80:
                            text = truncate_text(extract, max_words=120)
                            paragraphs.append(text)
                            break

                    time.sleep(1.0)

                time.sleep(2.0)
                break  # success, exit retry loop
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    wait = 10 * (attempt + 1)
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"    WARNING: {e}")
                    break
            except Exception as e:
                print(f"    WARNING: {e}")
                break

    return paragraphs


def main():
    parser = argparse.ArgumentParser(description="Fetch human texts from Wikipedia")
    parser.add_argument("--max-samples", type=int, default=180,
                        help="Maximum number of paragraphs to fetch")
    args = parser.parse_args()

    print(f"Fetching up to {args.max_samples} Wikipedia paragraphs...")
    paragraphs = fetch_wikipedia_paragraphs(max_samples=args.max_samples)

    if not paragraphs:
        print("ERROR: No paragraphs fetched.")
        sys.exit(1)

    df = pd.DataFrame({"Text": paragraphs, "Label": "Human"})

    avg_words = df["Text"].apply(lambda x: len(x.split())).mean()
    print(f"\nFetched {len(df)} paragraphs (avg {avg_words:.0f} words)")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, sep=";", index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
