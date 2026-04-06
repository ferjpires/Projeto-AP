import time
import random
from typing import Callable, Optional

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

VALID_LABELS  = ["Anthropic", "Google", "Human", "Meta", "OpenAI"]
SEED          = 42


def build_support_set(
    df: pd.DataFrame,
    n_per_class: int = 1,
    text_col: str = "Text",
    label_col: str = "Label",
    seed: int = SEED,
) -> list[dict]:
    rng = random.Random(seed)
    support = []

    for label in VALID_LABELS:
        subset = df[df[label_col] == label].copy()
        if subset.empty:
            continue
        subset["_wc"] = subset[text_col].str.split().str.len()
        median_wc = subset["_wc"].median()
        subset["_dist"] = (subset["_wc"] - median_wc).abs()
        candidates = subset.nsmallest(max(n_per_class * 3, 10), "_dist")
        chosen = candidates.sample(n=min(n_per_class, len(candidates)), random_state=seed)
        for _, row in chosen.iterrows():
            support.append({text_col: row[text_col], label_col: row[label_col]})

    rng.shuffle(support)
    return support


def run_experiment(
    df: pd.DataFrame,
    ask_fn: Callable[[str], str],
    prompt_builder: Callable,
    normalize_fn: Callable,
    support_examples: Optional[list] = None,
    text_col: str = "Text",
    sleep_seconds: float = 0.3,
    verbose: bool = True,
) -> tuple[list, list]:
    predictions = []
    raw_outputs = []

    for i, (_, row) in enumerate(df.iterrows()):
        text = row[text_col]
        try:
            if support_examples is not None:
                prompt = prompt_builder(text, support_examples)
            else:
                prompt = prompt_builder(text)

            raw = ask_fn(prompt)
            pred = normalize_fn(raw)

            raw_outputs.append(raw)
            predictions.append(pred)

            if verbose and (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(df)}] done")

            time.sleep(sleep_seconds)

        except Exception as e:
            print(f"  [!] Error on row {i}: {e}")
            raw_outputs.append(str(e))
            predictions.append(None)

    return predictions, raw_outputs


def evaluate(
    predictions: list,
    df: pd.DataFrame,
    label_col: str = "Label",
    experiment_name: str = "",
) -> pd.DataFrame:
    y_true = df[label_col].tolist()

    rows = []
    for pred, true in zip(predictions, y_true):
        rows.append({
            "true_label":  true,
            "prediction":  pred,
            "correct":     pred == true,
        })
    results_df = pd.DataFrame(rows)

    valid = results_df.dropna(subset=["prediction"])
    null_rate = results_df["prediction"].isna().mean()
    acc = (
        accuracy_score(valid["true_label"], valid["prediction"])
        if not valid.empty else 0.0
    )

    print(f"\n{'='*50}")
    if experiment_name:
        print(f"  {experiment_name}")
    print(f"  Accuracy : {acc:.4f}  ({len(valid)}/{len(results_df)} parseable)")
    print(f"  Null rate: {null_rate:.2%}")
    print(f"{'='*50}")
    if valid.empty:
        print("No parseable predictions.")
    else:
        print(classification_report(
            valid["true_label"], valid["prediction"],
            labels=VALID_LABELS, zero_division=0
        ))

    return results_df


def compare_experiments(experiment_dict: dict[str, list], df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    y_true = df[label_col].tolist()
    rows = []
    for name, preds in experiment_dict.items():
        valid_pairs = [(p, t) for p, t in zip(preds, y_true) if p is not None]
        if not valid_pairs:
            rows.append({"experiment": name, "accuracy": 0.0, "n_valid": 0})
            continue
        pv, tv = zip(*valid_pairs)
        acc = accuracy_score(tv, pv)
        rows.append({"experiment": name, "accuracy": acc, "n_valid": len(valid_pairs)})
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


def confusion_df(predictions: list, df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    y_true = df[label_col].tolist()
    valid = [(p, t) for p, t in zip(predictions, y_true) if p is not None]
    if not valid:
        cm = [[0] * len(VALID_LABELS) for _ in VALID_LABELS]
        return pd.DataFrame(cm, index=VALID_LABELS, columns=VALID_LABELS)

    pv, tv = zip(*valid)
    cm = confusion_matrix(tv, pv, labels=VALID_LABELS)
    return pd.DataFrame(cm, index=VALID_LABELS, columns=VALID_LABELS)


def predict_dataset(
    df: pd.DataFrame,
    ask_fn: Callable,
    prompt_builder: Callable,
    normalize_fn: Callable,
    support_examples: Optional[list] = None,
    text_col: str = "Text",
    sleep_seconds: float = 0.3,
) -> pd.DataFrame:
    preds, raws = run_experiment(
        df, ask_fn, prompt_builder, normalize_fn,
        support_examples=support_examples,
        text_col=text_col,
        sleep_seconds=sleep_seconds,
        verbose=True,
    )

    non_null = [p for p in preds if p is not None]
    fallback = max(set(non_null), key=non_null.count) if non_null else "Human"

    out = df.copy()
    out["predicted_label"] = [p if p is not None else fallback for p in preds]
    out["raw_output"]      = raws
    return out
