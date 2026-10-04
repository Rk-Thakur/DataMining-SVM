"""
SVM class assignment
--------------------
"Please train the testing dataset given in the class folder by SVM and create a
model. Use the model to predict the Training dataset given in the class folder.
Simply report the accuracy."

Usage:
    python svm_assignment.py                       # train on Testing, evaluate (LOOCV)
    python svm_assignment.py "Training dataset.xlsx"   # also predict the Training file

Data format (both files): column A = wine name, column B = class
(1: grade 90+, 0: below 90), remaining columns = 0/1 flavor-word attributes.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Assignment: build the model on the Testing set. Matches "Testing dataset.xlsx",
# "Testing dataset (1).xlsx", etc., and the path is resolved relative to this script.
HERE = Path(__file__).resolve().parent
_matches = sorted(HERE.glob("Testing*.xlsx"))
if not _matches:
    sys.exit(f"No Testing dataset (Testing*.xlsx) found in {HERE}")
TRAIN_FILE = _matches[0]


def load(path):
    df = pd.read_excel(path)
    df = df.rename(columns={df.columns[0]: "Wine", df.columns[1]: "Class"})
    X = df.drop(columns=["Wine", "Class"]).fillna(0).astype(float)
    y = df["Class"].astype(int)
    return df["Wine"], X, y


def write_svmlight(X, y, path):
    """Write the data in the SVM-light format shown in the slides (label +1/-1, sparse idx:val)."""
    with open(path, "w") as f:
        for row, label in zip(X.values, y.values):
            feats = " ".join(f"{i + 1}:{v:g}" for i, v in enumerate(row) if v != 0)
            f.write(f"{1 if label == 1 else -1} {feats}\n")


names, X, y = load(TRAIN_FILE)
print(f"Training data: {TRAIN_FILE.name}  ->  {X.shape[0]} wines, {X.shape[1]} attributes")
print(f"Class balance: 90+ = {(y == 1).sum()},  below 90 = {(y == 0).sum()}\n")

write_svmlight(X, y, "testing_svmlight.dat")

model = SVC(kernel="linear", C=1.0)
model.fit(X, y)
joblib.dump((model, list(X.columns)), "svm_model.joblib")
print(f"Linear SVM trained. Support vectors: {model.n_support_.sum()} "
      f"(class 0: {model.n_support_[0]}, class 1: {model.n_support_[1]})")

train_acc = accuracy_score(y, model.predict(X))
print(f"Accuracy on the data it was trained on: {train_acc:.2%}")

loo_pred = cross_val_predict(SVC(kernel="linear", C=1.0), X, y, cv=LeaveOneOut())
print(f"Leave-one-out cross-validation accuracy: {accuracy_score(y, loo_pred):.2%} "
      f"({(loo_pred == y).sum()}/{len(y)})")
print("LOOCV confusion matrix [rows = actual 0/1, cols = predicted 0/1]:")
print(confusion_matrix(y, loo_pred), "\n")

# Most influential words learned by the linear SVM
w = pd.Series(model.coef_[0], index=X.columns)
print("Top words pushing toward 90+  :", ", ".join(w.nlargest(8).index))
print("Top words pushing toward <90  :", ", ".join(w.nsmallest(8).index), "\n")

# Predict the Training dataset, if supplied
if len(sys.argv) > 1:
    if Path(sys.argv[1]).resolve() == TRAIN_FILE:
        print("Note: that is the same file the model was trained on, so the accuracy "
              "below is not a real test. Pass the Training dataset instead.\n")
    t_names, tX, ty = load(sys.argv[1])
    tX = tX.reindex(columns=X.columns, fill_value=0)   # align attributes with the model
    pred = model.predict(tX)
    acc = accuracy_score(ty, pred)
    print(f"=== Prediction on {sys.argv[1]} ({len(ty)} wines) ===")
    print(f"ACCURACY: {acc:.2%}  ({(pred == ty.values).sum()}/{len(ty)} correct)")
    print(confusion_matrix(ty, pred))
    pd.DataFrame({"Wine": t_names, "Actual": ty, "Predicted": pred}).to_csv(
        "training_predictions.csv", index=False)
    print("Per-wine predictions saved to training_predictions.csv")
else:
    print("No Training dataset supplied - run again with its path to get the assignment accuracy.")
