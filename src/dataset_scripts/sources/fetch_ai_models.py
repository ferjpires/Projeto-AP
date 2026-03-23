import os
import re
import time
import random
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TOPICS = [
    # Biology / Cell biology
    "The role of mitochondria in cellular respiration",
    "How DNA replication is initiated and carried out",
    "The function of ribosomes in protein synthesis",
    "How the immune system recognises foreign antigens",
    "The mechanism of CRISPR-Cas9 gene editing",
    "How neurons transmit electrical signals via action potentials",
    "The structure and function of the cell membrane",
    "How enzymes lower activation energy in biochemical reactions",
    "The process of meiosis and its role in genetic diversity",
    "How telomeres protect chromosomes from degradation",
    # Biomedicine / Health
    "How vaccines stimulate the adaptive immune response",
    "The pathophysiology of type 2 diabetes",
    "How PET scanning works and its clinical applications",
    "The role of vitamin D in bone metabolism and immune function",
    "How statins reduce LDL cholesterol levels",
    "The mechanism of action of mRNA vaccines",
    "How CRISPR is being applied to treat genetic diseases",
    "The difference between type 1 and type 2 diabetes at the molecular level",
    "How the blood-brain barrier regulates substance entry into the brain",
    "The role of gut microbiome in immune regulation",
    # Chemistry
    "How covalent bonds form between nonmetallic elements",
    "The principles of Le Chatelier in chemical equilibrium",
    "How catalysts accelerate chemical reactions without being consumed",
    "The structure of benzene and aromatic stability",
    "How electronegativity determines bond polarity",
    "The mechanism of acid-base reactions in aqueous solution",
    "How nuclear fission releases energy",
    "The principles behind chromatography as a separation technique",
    "How entropy relates to spontaneity in thermodynamics",
    "The role of oxidation states in redox reactions",
    # Physics
    "How thermonuclear fusion occurs in stars",
    "The photoelectric effect and its significance to quantum mechanics",
    "How black holes form and why light cannot escape them",
    "The principles of Bernoulli in fluid dynamics",
    "How superconductivity emerges at low temperatures",
    "The wave-particle duality of electrons",
    "How the Doppler effect applies to electromagnetic radiation",
    "The relationship between electric and magnetic fields in Maxwell's equations",
    "How semiconductors are used in transistors",
    "The principles of general relativity and spacetime curvature",
    # Mathematics
    "How the Fourier transform decomposes signals into frequencies",
    "The significance of Euler's identity in complex analysis",
    "How Bayesian inference updates probabilities with new evidence",
    "The concept of limits and their role in calculus",
    "How graph theory models networks and connectivity",
    "The proof strategy behind mathematical induction",
    "How prime numbers are distributed according to the prime number theorem",
    "The role of eigenvalues in linear transformations",
    "How Monte Carlo methods approximate complex integrals",
    "The significance of Gödel's incompleteness theorems",
    # Engineering / Technology
    "How lithium-ion batteries store and release energy",
    "The operating principle of a transistor in digital circuits",
    "How neural networks learn through backpropagation",
    "The principles behind OLED display technology",
    "How GPS triangulation determines geographic position",
    "The role of hash functions in cryptographic security",
    "How fiber optic cables transmit data using light",
    "The thermodynamic principles behind heat pump efficiency",
    "How RADAR detects objects using reflected radio waves",
    "The engineering principles behind suspension bridge stability",
]

SYSTEM_PROMPT = (
    "You are a scientific writer. For each topic given, write a single informative "
    "paragraph of 100-130 words. Be factual, use scientific language, no bullet points, "
    "no headers, no first-person. Number each response matching the topic number. "
    "Just the numbered paragraphs, nothing else."
)

VARIANT_SEEDS = [
    "Use a different sentence structure than you would normally.",
    "Vary your vocabulary and phrasing.",
    "Approach the explanation from a slightly different angle.",
]

VARIANTS_PER_TOPIC = 3

BATCH_PROMPT = """\
{seed}

For each topic below, write a single informative paragraph of 100-130 words. \
Be factual, use scientific language, no bullet points, no headers, no first-person. \
Number each response matching the topic number.

{numbered_topics}"""


def _parse_batch_response(text: str, n_topics: int) -> list:
    results = [""] * n_topics
    pattern = re.compile(
        r'(?:^|\n)\**(\d+)[.)]\**\.?\s+(.*?)(?=\n\**\d+[.)]\**\.?\s+|\Z)',
        re.DOTALL
    )
    for m in pattern.finditer(text):
        idx = int(m.group(1)) - 1
        if 0 <= idx < n_topics:
            para = m.group(2).strip().replace('\n', ' ')
            para = re.sub(r'\s+', ' ', para)
            results[idx] = para
    return results


def _groq_generate_batch(client, model: str, topics: list, variant_idx: int) -> list:
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(topics))
    user_msg = BATCH_PROMPT.format(
        seed=VARIANT_SEEDS[variant_idx % len(VARIANT_SEEDS)],
        numbered_topics=numbered,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=len(topics) * 220,
        temperature=0.85,
    )
    raw = response.choices[0].message.content.strip()
    return _parse_batch_response(raw, len(topics))


def _fetch_with_groq(
    groq_model: str,
    label: str,
    topics: list,
    variants: int,
    delay: float,
) -> pd.DataFrame:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Install groq: pip install groq")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    client = Groq(api_key=api_key)
    rows = []

    for v in range(variants):
        print(f"  [{label}] variant {v+1}/{variants} — sending {len(topics)} topics in one call...")
        try:
            paragraphs = _groq_generate_batch(client, groq_model, topics, v)
            ok = sum(1 for p in paragraphs if p)
            print(f"  [{label}] variant {v+1} — parsed {ok}/{len(topics)} paragraphs")
            for topic, para in zip(topics, paragraphs):
                if para:
                    rows.append({"Text": para, "Label": label})
                else:
                    print(f"    WARNING: empty response for topic '{topic[:60]}...'")
        except Exception as e:
            print(f"  [{label}] variant {v+1} ERROR: {e}")

        if v < variants - 1:
            time.sleep(delay + random.uniform(0, 1.0))

    df = pd.DataFrame(rows)
    print(f"  Total {label}: {len(df)} samples generated")
    return df


def _gemini_generate_batch(model, topics: list, variant_idx: int) -> list:
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(topics))
    prompt = (
        SYSTEM_PROMPT + "\n\n"
        + BATCH_PROMPT.format(
            seed=VARIANT_SEEDS[variant_idx % len(VARIANT_SEEDS)],
            numbered_topics=numbered,
        )
    )
    response = model.generate_content(prompt)
    return _parse_batch_response(response.text.strip(), len(topics))


def _fetch_google_gemini(topics: list, variants: int, delay: float) -> pd.DataFrame:
    try:
        from google import genai
    except ImportError:
        raise ImportError("Install google-genai: pip install google-genai")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)
    rows = []

    print("Fetching Google (gemini-2.5-flash) via Gemini API...")
    for v in range(variants):
        print(f"  [Google] variant {v+1}/{variants} — sending {len(topics)} topics in one call...")
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(topics))
        prompt = (
            SYSTEM_PROMPT + "\n\n"
            + BATCH_PROMPT.format(
                seed=VARIANT_SEEDS[v % len(VARIANT_SEEDS)],
                numbered_topics=numbered,
            )
        )
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            raw = response.text.strip()
            paragraphs = _parse_batch_response(raw, len(topics))
            ok = sum(1 for p in paragraphs if p)
            print(f"  [Google] variant {v+1} — parsed {ok}/{len(topics)} paragraphs")
            for topic, para in zip(topics, paragraphs):
                if para:
                    rows.append({"Text": para, "Label": "Google"})
                else:
                    print(f"    WARNING: empty response for '{topic[:50]}...'")
        except Exception as e:
            print(f"  [Google] variant {v+1} ERROR: {e}")

        if v < variants - 1:
            time.sleep(delay + random.uniform(0, 1.0))

    df = pd.DataFrame(rows)
    print(f"  Total Google: {len(df)} samples generated")
    return df


def fetch_meta(
    topics: list = None,
    variants: int = VARIANTS_PER_TOPIC,
    delay: float = 2.0,
) -> pd.DataFrame:
    print("Fetching Meta (llama-3.3-70b) via Groq...")
    return _fetch_with_groq(
        groq_model="llama-3.3-70b-versatile",
        label="Meta",
        topics=topics or TOPICS,
        variants=variants,
        delay=delay,
    )


def fetch_google(
    topics: list = None,
    variants: int = VARIANTS_PER_TOPIC,
    delay: float = 2.0,
) -> pd.DataFrame:
    topics = topics or TOPICS
    return _fetch_google_gemini(topics, variants, delay)


def fetch_openai_from_csv(csv_path: str) -> pd.DataFrame:
    print(f"Loading OpenAI texts from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        if "Label" not in df.columns:
            df["Label"] = "OpenAI"
        df = df[["Text", "Label"]].dropna()
        df = df[df["Label"] == "OpenAI"].copy()
        print(f"  Loaded {len(df)} OpenAI samples")
        return df
    except FileNotFoundError:
        print(f"  WARNING: {csv_path} not found — skipping OpenAI manual data")
        return pd.DataFrame(columns=["Text", "Label"])


def fetch_anthropic_from_csv(csv_path: str) -> pd.DataFrame:
    print(f"Loading Anthropic texts from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        if "Label" not in df.columns:
            df["Label"] = "Anthropic"
        df = df[["Text", "Label"]].dropna()
        df = df[df["Label"] == "Anthropic"].copy()
        print(f"  Loaded {len(df)} Anthropic samples")
        return df
    except FileNotFoundError:
        print(f"  WARNING: {csv_path} not found — skipping Anthropic manual data")
        return pd.DataFrame(columns=["Text", "Label"])