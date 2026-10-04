"""
Data Mining SVM assignment (SVMlight data).

Train a linear SVM on the file named "testing" and predict the file named
"Training", then report the accuracy.

    python svm_svmlight.py --train data/first_test_3_1.txt \
                           --evaluate data/first_train_3_1.txt
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC


def parse_args():
    p = argparse.ArgumentParser(description="Train a linear SVM on SVMlight data and report accuracy.")
    p.add_argument("--train", required=True, help="SVMlight file to train on (the 'testing' file)")
    p.add_argument("--evaluate", required=True, help="SVMlight file to predict (the 'Training' file)")
    p.add_argument("--model", default="output/svm_model.joblib", help="where to save the trained model")
    p.add_argument("--predictions", default="output/predictions.txt", help="where to save predicted labels")
    p.add_argument("--metrics", default="output/metrics.json", help="where to save the JSON metrics report")
    p.add_argument("--C", type=float, default=1.0, help="SVM regularisation parameter (default 1.0)")
    p.add_argument("--chunk-size", type=int, default=10000, help="evaluation rows read per chunk")
    return p.parse_args()


def to_int32(X):
    """LinearSVC only accepts sparse matrices with 32-bit indices."""
    X = csr_matrix(X)
    X.indices = X.indices.astype(np.int32)
    X.indptr = X.indptr.astype(np.int32)
    return X


def parse_chunk(lines, n_features):
    """Parse SVMlight lines into (X, y), keeping only non-zero values. Features beyond n_features are ignored."""
    labels, indptr, indices, data = [], [0], [], []
    for line in lines:
        parts = line.split("#", 1)[0].split()
        labels.append(float(parts[0]))
        for item in parts[1:]:
            idx, val = item.split(":")
            idx, val = int(idx) - 1, float(val)     # SVMlight indices are 1-based
            if val != 0 and idx < n_features:
                indices.append(idx)
                data.append(val)
        indptr.append(len(indices))
    X = csr_matrix((data, indices, indptr), shape=(len(labels), n_features))
    return to_int32(X), np.asarray(labels)


def iter_chunks(path, chunk_size, n_features):
    with open(path) as f:
        lines = []
        for line in f:
            if line.strip():
                lines.append(line)
            if len(lines) == chunk_size:
                yield parse_chunk(lines, n_features)
                lines = []
        if lines:
            yield parse_chunk(lines, n_features)


def main():
    args = parse_args()
    for out in (args.model, args.predictions, args.metrics):
        Path(out).parent.mkdir(parents=True, exist_ok=True)

    # Train
    t0 = time.time()
    X_train, y_train = load_svmlight_file(args.train)
    X_train.eliminate_zeros()                     # drop the explicit "i:0" entries
    X_train = to_int32(X_train)
    n_features = X_train.shape[1]
    counts = {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}
    print(f"Training file : {args.train}")
    print(f"  {X_train.shape[0]:,} examples, {n_features:,} features, labels {counts}")

    model = LinearSVC(C=args.C, max_iter=10000)
    model.fit(X_train, y_train)
    joblib.dump(model, args.model)
    print(f"  trained LinearSVC (C={args.C}) in {time.time() - t0:.1f}s -> {args.model}")
    train_acc = accuracy_score(y_train, model.predict(X_train))

    # Evaluate in chunks so the large file never sits in memory all at once
    t0 = time.time()
    y_true, y_pred = [], []
    with open(args.predictions, "w") as out:
        for X, y in iter_chunks(args.evaluate, args.chunk_size, n_features):
            pred = model.predict(X)
            out.writelines(f"{int(p)}\n" for p in pred)
            y_true.append(y)
            y_pred.append(pred)
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

    correct = int((y_true == y_pred).sum())
    acc = correct / len(y_true)
    labels = [int(l) for l in sorted(np.unique(np.concatenate([y_true, y_pred])))]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\nEvaluation file : {args.evaluate}")
    print(f"  predicted {len(y_true):,} examples in {time.time() - t0:.1f}s -> {args.predictions}")
    print(f"  confusion matrix (rows = actual {labels}, cols = predicted):\n{cm}")
    print(f"\nACCURACY: {acc:.2%}  ({correct:,} / {len(y_true):,} correct)")

    metrics = {
        "train_file": args.train,
        "evaluate_file": args.evaluate,
        "model": f"LinearSVC(C={args.C})",
        "train_examples": int(X_train.shape[0]),
        "features": int(n_features),
        "train_accuracy": train_acc,
        "evaluate_examples": int(len(y_true)),
        "correct": correct,
        "accuracy": acc,
        "confusion_matrix": {"labels": [int(l) for l in labels], "matrix": cm.tolist()},
    }
    Path(args.metrics).write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {args.metrics}")


if __name__ == "__main__":
    main()
