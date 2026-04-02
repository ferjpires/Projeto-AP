import os
import sys
import json
import time
import random
import argparse
import urllib.request
import urllib.parse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_RAW_GENERATED

OUTPUT_PATH = os.path.join(DATA_RAW_GENERATED, "human.csv")
HEADERS = {"User-Agent": "StyleDetectionResearch/1.0 (academic project)"}

WIKI_CATEGORIES = [
    "Biology", "Medicine", "Microbiology", "Ecology",
    "Chemistry", "Organic_chemistry",
    "Physics", "Quantum_mechanics",
    "Mathematics", "Statistics",
    "Computer_science", "Engineering", "Science"
]


def get_category_members(category, limit=500):
    url = (
        "https://en.wikipedia.org/w/api.php?"
        + urllib.parse.urlencode({
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": str(limit),
            "cmtype": "page",
            "format": "json",
        })
    )
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        members = data.get("query", {}).get("categorymembers", [])
        return [m["title"] for m in members]
    except Exception as e:
        print(f"  WARNING: Failed to get category {category}: {e}")
        return []


def get_wikipedia_extract(title):
    url = (
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
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "")
    return ""


def truncate_text(text, target_words=110):
    words = text.split()
    if len(words) <= target_words:
        return text
    truncated = " ".join(words[:target_words])
    last_dot = truncated.rfind(".")
    if last_dot != -1:
        truncated = truncated[:last_dot + 1]
    return truncated


def load_existing_texts(path):
    if not os.path.exists(path):
        return set(), pd.DataFrame(columns=["Text", "Label"])
    try:
        df = pd.read_csv(path, sep=";")
        return set(df["Text"].tolist()), df
    except Exception:
        return set(), pd.DataFrame(columns=["Text", "Label"])


def fetch_from_categories(target_new, seen_texts):
    paragraphs = []
    seen_titles = set()

    # Gather all candidate articles from all categories
    all_articles = []
    for cat in WIKI_CATEGORIES:
        print(f"  Loading category: {cat}...")
        members = get_category_members(cat)
        for title in members:
            all_articles.append((cat, title))
        time.sleep(random.uniform(2.0, 3.0))

    print(f"  Total candidate articles: {len(all_articles)}")
    random.shuffle(all_articles)

    for cat, title in all_articles:
        if len(paragraphs) >= target_new:
            break

        if title in seen_titles:
            continue
        seen_titles.add(title)

        for attempt in range(3):
            try:
                extract = get_wikipedia_extract(title)
                words = extract.split()
                if len(words) >= 80:
                    text = truncate_text(extract, target_words=110)
                    # Skip if too similar to existing
                    if text not in seen_texts and len(text.split()) >= 60:
                        paragraphs.append(text)
                        seen_texts.add(text)
                        if len(paragraphs) % 25 == 0:
                            print(f"  Fetched {len(paragraphs)}/{target_new}...")
                time.sleep(0.5)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    wait = 10 * (attempt + 1)
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    break
            except Exception:
                break

    return paragraphs


def main():
    parser = argparse.ArgumentParser(description="Fetch human texts from Wikipedia categories")
    parser.add_argument("--target", type=int, default=500,
                        help="Total target number of human samples (existing + new)")
    args = parser.parse_args()

    print(f"Target: {args.target} total human samples")

    existing_texts, existing_df = load_existing_texts(OUTPUT_PATH)
    current_count = len(existing_df)
    print(f"Existing: {current_count} samples")

    needed = max(0, args.target - current_count)
    if needed == 0:
        print("Already have enough samples!")
        return

    print(f"Need {needed} more samples")
    new_paragraphs = fetch_from_categories(needed, existing_texts)

    if not new_paragraphs:
        print("ERROR: No new paragraphs fetched.")
        sys.exit(1)

    new_df = pd.DataFrame({"Text": new_paragraphs, "Label": "Human"})
    combined = pd.concat([existing_df, new_df], ignore_index=True)

    avg_words = combined["Text"].apply(lambda x: len(x.split())).mean()
    print(f"\nTotal: {len(combined)} paragraphs (avg {avg_words:.0f} words)")
    print(f"  Existing: {current_count}, New: {len(new_paragraphs)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    combined.to_csv(OUTPUT_PATH, sep=";", index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
