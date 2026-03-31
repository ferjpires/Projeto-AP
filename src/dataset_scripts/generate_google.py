import os
import sys
import time
import random
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    TOPICS, BATCH_PROMPT,
    DATA_RAW_GENERATED, load_env, parse_batch_response, append_to_csv,
)

OUTPUT_PATH = os.path.join(DATA_RAW_GENERATED, "google.csv")


def generate_batch(client, topics):
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(topics))
    prompt = BATCH_PROMPT.format(numbered_topics=numbered)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return parse_batch_response(response.text.strip(), len(topics))


def main():
    parser = argparse.ArgumentParser(description="Generate Google texts via Gemini API")
    parser.add_argument("--variants", type=int, default=50, help="Number of variants to generate per topic")
    parser.add_argument("--batch-size", type=int, default=10, help="Topics per API call")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between batches")
    args = parser.parse_args()

    keys = load_env()
    if not keys["GEMINI_API_KEY"]:
        print("ERROR: GEMINI_API_KEY not found in .env")
        sys.exit(1)

    try:
        from google import genai
    except ImportError:
        print("ERROR: Install google-genai: pip install google-genai")
        sys.exit(1)

    client = genai.Client(api_key=keys["GEMINI_API_KEY"])

    total_generated = 0
    for v in range(args.variants):
        # Process topics in batches
        for start in range(0, len(TOPICS), args.batch_size):
            batch = TOPICS[start:start + args.batch_size]
            print(f"  [Google] variant {v+1}/{args.variants}, "
                  f"batch {start//args.batch_size + 1} ({len(batch)} topics)...")

            try:
                paragraphs = generate_batch(client, batch)
                rows = []
                for topic, para in zip(batch, paragraphs):
                    if para:
                        rows.append({"Text": para, "Label": "Google"})
                    else:
                        print(f"    WARNING: empty for '{topic[:50]}...'")

                if rows:
                    df = pd.DataFrame(rows)
                    append_to_csv(df, OUTPUT_PATH)
                    total_generated += len(rows)
            except Exception as e:
                print(f"    ERROR: {e}")

            time.sleep(args.delay + random.uniform(0, 1.0))

    print(f"\nDone. Generated {total_generated} Google paragraphs total.")


if __name__ == "__main__":
    main()
