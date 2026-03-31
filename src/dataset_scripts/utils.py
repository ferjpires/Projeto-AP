import os
import re
import csv
import pandas as pd
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_RAW_GENERATED = os.path.join(PROJECT_ROOT, "data", "raw", "generated")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")

TOPICS = [
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

# Wikipedia search terms (shorter, for API queries)
SCIENTIFIC_SEARCH_TERMS = [
    "mitochondria cellular respiration", "DNA replication",
    "ribosomes protein synthesis", "immune system antigens",
    "CRISPR-Cas9 gene editing", "neurons action potentials",
    "cell membrane structure", "enzymes activation energy",
    "meiosis genetic diversity", "telomeres chromosomes",
    "vaccines immune response", "type 2 diabetes",
    "PET scanning", "vitamin D bone metabolism",
    "statins cholesterol", "mRNA vaccines",
    "CRISPR genetic diseases", "blood-brain barrier",
    "gut microbiome immune", "covalent bonds",
    "Le Chatelier equilibrium", "catalysts chemical reactions",
    "benzene aromatic stability", "electronegativity bond polarity",
    "acid-base reactions", "nuclear fission energy",
    "chromatography separation", "entropy thermodynamics",
    "oxidation states redox", "thermonuclear fusion stars",
    "photoelectric effect quantum", "black holes light",
    "Bernoulli fluid dynamics", "superconductivity",
    "wave-particle duality electrons", "Doppler effect electromagnetic",
    "Maxwell equations electric magnetic", "semiconductors transistors",
    "general relativity spacetime", "Fourier transform frequencies",
    "Euler identity complex analysis", "Bayesian inference probability",
    "limits calculus", "graph theory networks",
    "mathematical induction proof", "prime numbers distribution",
    "eigenvalues linear transformations", "Monte Carlo methods integrals",
    "Gödel incompleteness theorems", "lithium-ion batteries energy",
    "transistor digital circuits", "neural networks backpropagation",
    "OLED display technology", "GPS triangulation",
    "hash functions cryptographic", "fiber optic cables light",
    "heat pump thermodynamic", "RADAR radio waves",
    "suspension bridge stability",
]

BATCH_PROMPT = """\
For each topic below, write a single paragraph of EXACTLY 110 to 130 words. \
Number each response matching the topic number.

{numbered_topics}"""


def load_env():
    env_path = os.path.join(PROJECT_ROOT, "src", ".env")
    load_dotenv(env_path)
    return {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    }


def parse_batch_response(text: str, n_topics: int) -> list:
    results = [""] * n_topics
    pattern = re.compile(
        r'(?:^|\n)\**(\d+)[.)]\**\.?\s+(.*?)(?=\n\**\d+[.)]\**\.?\s+|\Z)',
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        idx = int(m.group(1)) - 1
        if 0 <= idx < n_topics:
            para = m.group(2).strip().replace("\n", " ")
            para = re.sub(r"\s+", " ", para)
            results[idx] = para
    return results


def truncate_text(text: str, max_words: int = 120) -> str:
    return " ".join(str(text).split()[:max_words])


def balance_classes(
    df, label_col="Label", random_state=42, max_per_class=500
) -> pd.DataFrame:
    sampled = [
        g.sample(n=min(len(g), max_per_class), random_state=random_state)
        for _, g in df.groupby(label_col)
    ]
    return pd.concat(sampled, ignore_index=True)


def deduplicate(df, text_col="Text") -> pd.DataFrame:
    return df.drop_duplicates(subset=[text_col]).reset_index(drop=True)


def save_dataset(df, path, prefix="DATASET"):
    df = df.copy()
    df["ID"] = [f"{prefix}-{i + 1}" for i in range(len(df))]
    df_final = df[["ID", "Text", "Label"]]
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df_final.to_csv(path, sep=";", index=False, quoting=csv.QUOTE_ALL)
    print(f"Saved {len(df_final)} samples to {path}")
    return df_final


def append_to_csv(df, path, sep=";"):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    df.to_csv(path, mode="a", sep=sep, index=False, header=write_header)
    print(f"  Appended {len(df)} rows to {path}")


def load_generated_csv(path):
    with open(path, "r") as f:
        header = f.readline().strip()

    sep = ";" if ";" in header else ","

    df = pd.read_csv(path, sep=sep, on_bad_lines="skip")
    if "Text" not in df.columns or "Label" not in df.columns:
        raise ValueError(f"{path}: expected 'Text' and 'Label' columns, got {df.columns.tolist()}")
    return df[["Text", "Label"]].dropna()
