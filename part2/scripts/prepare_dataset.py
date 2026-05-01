"""
Organiza as imagens de processed/ em train/val/test por classe,
usando o metadata.csv do dataset MIQR-CC.

Uso:
    python scripts/prepare_dataset.py

Antes de correr:
    - Mete a pasta processed/ em data/processed/ (com as imagens todas soltas)
    - Mete o metadata.csv em data/metadata.csv
"""

import pandas as pd
import shutil
from pathlib import Path
import sys

# ── Configuração ──────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_CSV  = PROJECT_ROOT / "data" / "metadata.csv"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "processed"

CLASS_NAMES = ["Biliary_Leaks", "Lithiasis", "Normal", "Stricture"]
SPLITS      = ["train", "val", "test"]

# Mapeamento exacto das labels do CSV para os nomes de classe do projecto
LABEL_MAP = {
    "Biliary Leaks":       "Biliary_Leaks",
    "Lithiasis":           "Lithiasis",
    "Normal":              "Normal",
    "Benign Stricture":    "Stricture",
    "Malignant Stricture": "Stricture",
}
# ──────────────────────────────────────────────────────────────


def main():
    if not METADATA_CSV.exists():
        print(f"[ERRO] Nao encontrei: {METADATA_CSV}")
        sys.exit(1)

    # 1. Ler CSV
    df = pd.read_csv(METADATA_CSV)
    print(f"Total de linhas: {len(df)}")

    # 2. Filtrar Keep == "Keep"
    df = df[df["Keep"] == "Keep"].copy()
    print(f"Apos filtro Keep:    {len(df)}")

    # 3. Filtrar as 4 classes (ignorar Unlabelled)
    df = df[df["Label"].isin(LABEL_MAP.keys())].copy()
    df["class_name"] = df["Label"].map(LABEL_MAP)
    print(f"Apos filtro classes: {len(df)}")

    print("\nDistribuicao por classe:")
    print(df["class_name"].value_counts().to_string())

    # 4. Splits estratificados 70/15/15
    print("\nA criar splits 70/15/15...")
    df = _make_splits(df, "class_name")
    print(df.groupby(["split", "class_name"]).size().to_string())

    # 5. Criar pastas
    for split in SPLITS:
        for cls in CLASS_NAMES:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    # 6. Copiar imagens
    copied, missing = 0, 0
    for _, row in df.iterrows():
        fname      = Path(row["processed_image_path"]).name
        cls_name   = row["class_name"]
        split_name = row["split"]

        src = PROCESSED_DIR / fname
        if not src.exists():
            missing += 1
            if missing <= 5:
                print(f"  [AVISO] Nao encontrado: {fname}")
            continue

        dest = OUTPUT_DIR / split_name / cls_name / fname
        if not dest.exists():
            shutil.copy2(src, dest)
        copied += 1

    if missing > 5:
        print(f"  ... e mais {missing - 5} ficheiros em falta.")

    print(f"\nConcluido!  Copiados: {copied}  |  Em falta: {missing}")

    # 7. Resumo final
    print("\nDistribuicao final:")
    for split in SPLITS:
        print(f"\n  {split.upper()}:")
        for cls in CLASS_NAMES:
            d = OUTPUT_DIR / split / cls
            n = sum(1 for f in d.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"})
            print(f"    {cls:<20}: {n}")


def _make_splits(df, label_col, train_ratio=0.70, val_ratio=0.15):
    from sklearn.model_selection import train_test_split
    parts = []
    for cls in df[label_col].unique():
        sub = df[df[label_col] == cls].copy()
        train, temp = train_test_split(sub, test_size=1 - train_ratio,
                                       random_state=42, shuffle=True)
        val_frac = val_ratio / (1 - train_ratio)
        val, test = train_test_split(temp, test_size=1 - val_frac,
                                     random_state=42, shuffle=True)
        train["split"] = "train"
        val["split"]   = "val"
        test["split"]  = "test"
        parts.extend([train, val, test])
    return pd.concat(parts).reset_index(drop=True)


if __name__ == "__main__":
    main()