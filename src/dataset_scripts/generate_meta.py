import os
import sys
import time
import random
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    TOPICS, SYSTEM_PROMPT, BATCH_PROMPT, VARIANT_SEEDS,
    DATA_RAW_GENERATED, load_env, parse_batch_response, append_to_csv,
)

OUTPUT_PATH = os.path.join(DATA_RAW_GENERATED, "meta.csv")
MODEL = "llama-3.3-70b-versatile"


def generate_batch(client, topics, variant_idx):
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(topics))
    user_msg = BATCH_PROMPT.format(
        seed=VARIANT_SEEDS[variant_idx % len(VARIANT_SEEDS)],
        numbered_topics=numbered,
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=len(topics) * 220,
        temperature=0.85,
    )
    raw = response.choices[0].message.content.strip()
    return parse_batch_response(raw, len(topics))


def main():
    parser = argparse.ArgumentParser(description="Generate Meta texts via Groq API")
    parser.add_argument("--variants", type=int, default=50, help="Number of variants to generate per topic")
    parser.add_argument("--batch-size", type=int, default=10, help="Topics per API call")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between batches")
    args = parser.parse_args()

    keys = load_env()
    if not keys["GROQ_API_KEY"]:
        print("ERROR: GROQ_API_KEY not found in .env")
        sys.exit(1)

    try:
        from groq import Groq
    except ImportError:
        print("ERROR: Install groq: pip install groq")
        sys.exit(1)

    client = Groq(api_key=keys["GROQ_API_KEY"])

    total_generated = 0
    for v in range(args.variants):
        for start in range(0, len(TOPICS), args.batch_size):
            batch = TOPICS[start:start + args.batch_size]
            print(f"  [Meta] variant {v+1}/{args.variants}, "
                  f"batch {start//args.batch_size + 1} ({len(batch)} topics)...")

            try:
                paragraphs = generate_batch(client, batch, v)
                rows = []
                for topic, para in zip(batch, paragraphs):
                    if para:
                        rows.append({"Text": para, "Label": "Meta"})
                    else:
                        print(f"    WARNING: empty for '{topic[:50]}...'")

                if rows:
                    df = pd.DataFrame(rows)
                    append_to_csv(df, OUTPUT_PATH)
                    total_generated += len(rows)
            except Exception as e:
                print(f"    ERROR: {e}")

            time.sleep(args.delay + random.uniform(0, 1.0))

    print(f"\nDone. Generated {total_generated} Meta paragraphs total.")


if __name__ == "__main__":
    main()
