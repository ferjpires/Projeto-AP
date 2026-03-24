"""
Autonomous training and evaluation script.
Trains DNN (numpy), LSTM, and DistilBERT on the combined dataset,
evaluates against both validation sets, and reports results.

Usage:
    python scripts/train_and_evaluate.py
    python scripts/train_and_evaluate.py --skip-bert    # Skip DistilBERT (slow on CPU)
    python scripts/train_and_evaluate.py --dnn-only     # Only train DNN
"""

import sys
import os
import time
import random
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import nltk
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

from sklearn.model_selection import train_test_split

from src.data_processing import clean_text
from src.vectorizer import create_vectorizer
from src.models_numpy.dnn.neuralnet import NeuralNetwork
from src.models_numpy.dnn.layers import DenseLayer, DropoutLayer, BatchNormalizationLayer
from src.models_numpy.dnn.activation import ReLUActivation, SoftmaxActivation
from src.models_numpy.dnn.losses import CategoricalCrossEntropy
from src.models_numpy.dnn.metrics import accuracy
from src.models_numpy.dnn.optimizer import AdamOptimizer
from src.models_numpy.dnn.dataset import Dataset


# ============================================================
# Data loading
# ============================================================

CLASSES = ["Anthropic", "Google", "Human", "Meta", "OpenAI"]
LABEL_MAP = {label: i for i, label in enumerate(CLASSES)}

TRAIN_PATH = os.path.join(ROOT, "data", "processed", "dataset_combined.csv")
VAL1_PATH = os.path.join(ROOT, "data", "validation", "dataset-exemplos.csv")
VAL2_PATH = os.path.join(ROOT, "data", "validation", "subm1_labels_revealed.csv")


def load_data(path, sep=";"):
    df = pd.read_csv(path, sep=sep)
    df = df[df["Label"].isin(CLASSES)].copy()
    df["label_id"] = df["Label"].map(LABEL_MAP)
    df["text_clean"] = df["Text"].apply(clean_text)
    return df


def one_hot(labels, num_classes=5):
    oh = np.zeros((len(labels), num_classes))
    for i, l in enumerate(labels):
        oh[i, l] = 1
    return oh


# ============================================================
# DNN Training
# ============================================================

def build_dnn(input_dim, hidden_layers, dropout, num_classes, lr):
    net = NeuralNetwork(
        epochs=150,
        batch_size=64,
        optimizer=AdamOptimizer(learning_rate=lr),
        loss=CategoricalCrossEntropy,
        metric=accuracy,
        early_stopping=True,
        patience=15,
        verbose=False,
    )
    prev_dim = input_dim
    for h in hidden_layers:
        net.add(DenseLayer(prev_dim, h))
        net.add(BatchNormalizationLayer())
        net.add(ReLUActivation())
        net.add(DropoutLayer(dropout))
        prev_dim = h
    net.add(DenseLayer(prev_dim, num_classes))
    net.add(SoftmaxActivation())
    return net


def evaluate_dnn(model, vectorizer, df_val, needs_raw=False):
    texts_clean = list(df_val["text_clean"])
    texts_raw = list(df_val["Text"])
    labels = df_val["label_id"].values

    if needs_raw:
        X = vectorizer.transform(texts_clean, texts_raw)
    else:
        X = vectorizer.transform(texts_clean)

    y_oh = one_hot(labels)
    ds = Dataset(X, y_oh)
    preds = model.predict(ds)
    return accuracy(y_oh, preds)


def train_dnn_experiment(vectorizer_type, X_train, y_train_oh, X_val, y_val_oh,
                         hidden_layers, dropout, lr):
    input_dim = X_train.shape[1]
    net = build_dnn(input_dim, hidden_layers, dropout, 5, lr)

    ds_train = Dataset(X_train, y_train_oh)
    ds_val = Dataset(X_val, y_val_oh)

    history = net.fit(ds_train, ds_val)

    # Get best validation accuracy
    val_acc = max(history.get("val_acc", [0]))
    train_acc = max(history.get("train_acc", [0]))

    return net, val_acc, train_acc


def run_dnn_experiments(df_train, df_val1, df_val2, n_iterations=15):
    print("\n" + "=" * 60)
    print("DNN EXPERIMENTS (numpy)")
    print("=" * 60)

    vectorizer_configs = [
        ("forensic", True),
        ("stylometric", True),
        ("stylometric_only", True),
    ]

    all_results = []

    for vtype, needs_raw in vectorizer_configs:
        print(f"\n--- Vectorizer: {vtype} ---")

        # Create and fit vectorizer
        vec = create_vectorizer(vtype, max_words=1000)
        texts_clean = list(df_train["text_clean"])
        texts_raw = list(df_train["Text"])

        t0 = time.time()
        if needs_raw:
            X_all = vec.fit_transform(texts_clean, texts_raw)
        else:
            X_all = vec.fit_transform(texts_clean)
        print(f"  Vectorized: {X_all.shape} in {time.time()-t0:.1f}s")

        y_all = df_train["label_id"].values
        y_all_oh = one_hot(y_all)

        # Split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all_oh, test_size=0.2, random_state=42, stratify=y_all
        )

        # Random search
        configs = []
        for _ in range(n_iterations):
            configs.append({
                "hidden_layers": random.choice([[128, 64], [256, 128], [128, 64, 32], [64, 32]]),
                "dropout": random.choice([0.3, 0.4, 0.5]),
                "lr": random.choice([0.001, 0.005, 0.01]),
            })

        best_val = -1
        best_net = None

        for i, cfg in enumerate(configs):
            t0 = time.time()
            net, val_acc, train_acc = train_dnn_experiment(
                vtype, X_tr, y_tr, X_te, y_te,
                cfg["hidden_layers"], cfg["dropout"], cfg["lr"],
            )
            elapsed = time.time() - t0

            print(f"  [{i+1}/{n_iterations}] h={cfg['hidden_layers']} "
                  f"d={cfg['dropout']} lr={cfg['lr']} → "
                  f"train={train_acc:.4f} val={val_acc:.4f} ({elapsed:.1f}s)")

            if val_acc > best_val:
                best_val = val_acc
                best_net = net
                best_cfg = cfg

        # Evaluate best on external validation sets
        val1_acc = evaluate_dnn(best_net, vec, df_val1, needs_raw)
        val2_acc = evaluate_dnn(best_net, vec, df_val2, needs_raw)

        result = {
            "model": "DNN",
            "vectorizer": vtype,
            "best_config": str(best_cfg),
            "internal_val": best_val,
            "val1_acc": val1_acc,
            "val2_acc": val2_acc,
        }
        all_results.append(result)
        print(f"  BEST: internal={best_val:.4f} | val1={val1_acc:.4f} | val2={val2_acc:.4f}")

    return all_results


# ============================================================
# LSTM Training
# ============================================================

def run_lstm_experiments(df_train, df_val1, df_val2):
    print("\n" + "=" * 60)
    print("LSTM EXPERIMENTS (PyTorch)")
    print("=" * 60)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from src.features import Vocabulary, texts_to_sequences, TextDataset
    from src.models_pytorch.lstm import LSTMClassifier

    device = torch.device("cpu")
    results = []

    texts = list(df_train["text_clean"])
    labels = df_train["label_id"].values

    # Build vocabulary
    vocab = Vocabulary(max_words=10000)
    vocab.fit(texts)

    # Split
    tr_texts, te_texts, tr_labels, te_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    configs = [
        {"hidden_dim": 128, "n_layers": 2, "dropout": 0.3, "pooling": "attention", "lr": 0.001},
        {"hidden_dim": 64, "n_layers": 2, "dropout": 0.3, "pooling": "attention", "lr": 0.001},
        {"hidden_dim": 128, "n_layers": 1, "dropout": 0.3, "pooling": "mean", "lr": 0.001},
        {"hidden_dim": 64, "n_layers": 1, "dropout": 0.5, "pooling": "attention", "lr": 0.002},
    ]

    for ci, cfg in enumerate(configs):
        print(f"\n  [{ci+1}/{len(configs)}] LSTM: h={cfg['hidden_dim']} "
              f"layers={cfg['n_layers']} pool={cfg['pooling']}")

        max_len = 150
        X_tr = texts_to_sequences(tr_texts, vocab, max_len)
        X_te = texts_to_sequences(te_texts, vocab, max_len)

        train_ds = TextDataset(X_tr, tr_labels)
        val_ds = TextDataset(X_te, te_labels)
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=64)

        model = LSTMClassifier(
            vocab_size=len(vocab),
            embedding_dim=100,
            hidden_dim=cfg["hidden_dim"],
            output_dim=5,
            n_layers=cfg["n_layers"],
            dropout=cfg["dropout"],
            pooling=cfg["pooling"],
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0
        patience_counter = 0

        for epoch in range(30):
            model.train()
            for seqs, labs in train_dl:
                seqs, labs = seqs.to(device), labs.to(device)
                optimizer.zero_grad()
                out = model(seqs)
                loss = criterion(out, labs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validate
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for seqs, labs in val_dl:
                    seqs, labs = seqs.to(device), labs.to(device)
                    out = model(seqs)
                    preds = out.argmax(dim=1)
                    correct += (preds == labs).sum().item()
                    total += labs.size(0)

            val_acc = correct / total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

        model.load_state_dict(best_state)
        print(f"    Internal val: {best_val_acc:.4f}")

        # External validation
        def eval_lstm(df_ext):
            texts_ext = list(df_ext["text_clean"])
            labels_ext = df_ext["label_id"].values
            X_ext = texts_to_sequences(texts_ext, vocab, max_len)
            ext_ds = TextDataset(X_ext, labels_ext)
            ext_dl = DataLoader(ext_ds, batch_size=64)
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for seqs, labs in ext_dl:
                    out = model(seqs)
                    preds = out.argmax(dim=1)
                    correct += (preds == labs).sum().item()
                    total += labs.size(0)
            return correct / total

        v1 = eval_lstm(df_val1)
        v2 = eval_lstm(df_val2)
        print(f"    val1={v1:.4f} | val2={v2:.4f}")

        results.append({
            "model": f"LSTM ({cfg['pooling']})",
            "vectorizer": "sequences",
            "best_config": str(cfg),
            "internal_val": best_val_acc,
            "val1_acc": v1,
            "val2_acc": v2,
        })

    return results


# ============================================================
# DistilBERT Training
# ============================================================

def run_bert_experiments(df_train, df_val1, df_val2, epochs=3, batch_size=16):
    print("\n" + "=" * 60)
    print("DistilBERT EXPERIMENTS (PyTorch)")
    print("=" * 60)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from src.models_pytorch.distilbert import DistilBERTClassifier, DistilBERTDataset, get_tokenizer

    device = torch.device("cpu")

    tokenizer = get_tokenizer()
    texts = list(df_train["Text"])  # Raw text for BERT
    labels = df_train["label_id"].values

    tr_texts, te_texts, tr_labels, te_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_ds = DistilBERTDataset(tr_texts, tr_labels, tokenizer, max_len=128)
    val_ds = DistilBERTDataset(te_texts, te_labels, tokenizer, max_len=128)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = DistilBERTClassifier(output_dim=5, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_dl) * epochs
    print(f"  Training: {epochs} epochs, {len(train_dl)} batches/epoch, "
          f"{total_steps} total steps (CPU — this will take a while)")

    best_val_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        t0 = time.time()

        for batch_i, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labs = batch["label"].to(device)

            optimizer.zero_grad()
            out = model(input_ids, attention_mask)
            loss = criterion(out, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

            if (batch_i + 1) % 20 == 0:
                print(f"    Epoch {epoch+1} batch {batch_i+1}/{len(train_dl)} "
                      f"loss={loss.item():.4f}")

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labs = batch["label"].to(device)
                out = model(input_ids, attention_mask)
                preds = out.argmax(dim=1)
                correct += (preds == labs).sum().item()
                total += labs.size(0)

        val_acc = correct / total
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{epochs}: loss={train_loss/len(train_dl):.4f} "
              f"val_acc={val_acc:.4f} ({elapsed:.0f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # External validation
    def eval_bert(df_ext):
        ext_ds = DistilBERTDataset(list(df_ext["Text"]), df_ext["label_id"].values,
                                   tokenizer, max_len=128)
        ext_dl = DataLoader(ext_ds, batch_size=batch_size)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in ext_dl:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labs = batch["label"].to(device)
                out = model(input_ids, attention_mask)
                preds = out.argmax(dim=1)
                correct += (preds == labs).sum().item()
                total += labs.size(0)
        return correct / total

    v1 = eval_bert(df_val1)
    v2 = eval_bert(df_val2)
    print(f"  BEST: internal={best_val_acc:.4f} | val1={v1:.4f} | val2={v2:.4f}")

    return [{
        "model": "DistilBERT",
        "vectorizer": "tokenizer",
        "best_config": f"lr=2e-5, epochs={epochs}, bs={batch_size}",
        "internal_val": best_val_acc,
        "val1_acc": v1,
        "val2_acc": v2,
    }]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate all models")
    parser.add_argument("--skip-bert", action="store_true", help="Skip DistilBERT (slow)")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM")
    parser.add_argument("--dnn-only", action="store_true", help="Only train DNN")
    parser.add_argument("--dnn-iterations", type=int, default=15, help="DNN random search iterations")
    parser.add_argument("--bert-epochs", type=int, default=3, help="DistilBERT epochs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    print("Loading data...")
    df_train = load_data(TRAIN_PATH)
    df_val1 = load_data(VAL1_PATH)
    df_val2 = load_data(VAL2_PATH)

    print(f"  Train: {len(df_train)} samples")
    print(f"  Val1 (exemplos): {len(df_val1)} samples")
    print(f"  Val2 (subm1): {len(df_val2)} samples")
    print(f"  Classes: {CLASSES}")

    all_results = []

    # DNN
    dnn_results = run_dnn_experiments(df_train, df_val1, df_val2, n_iterations=args.dnn_iterations)
    all_results.extend(dnn_results)

    # LSTM
    if not args.dnn_only and not args.skip_lstm:
        lstm_results = run_lstm_experiments(df_train, df_val1, df_val2)
        all_results.extend(lstm_results)

    # DistilBERT
    if not args.dnn_only and not args.skip_bert:
        bert_results = run_bert_experiments(df_train, df_val1, df_val2,
                                            epochs=args.bert_epochs)
        all_results.extend(bert_results)

    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    results_df = pd.DataFrame(all_results)
    print(results_df[["model", "vectorizer", "internal_val", "val1_acc", "val2_acc"]].to_string(index=False))

    # Save results
    results_path = os.path.join(ROOT, "data", "processed", "experiment_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Best model
    best = results_df.loc[(results_df["val1_acc"] + results_df["val2_acc"]).idxmax()]
    print(f"\nBEST OVERALL: {best['model']} ({best['vectorizer']})")
    print(f"  val1={best['val1_acc']:.4f} val2={best['val2_acc']:.4f}")


if __name__ == "__main__":
    main()
