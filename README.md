# Data Mining – SVM Assignment

**Task:** Train an SVM on the *testing* dataset, use the model to predict the *Training* dataset, and report the accuracy.

## Data
From `SVM.rar` (SVMlight format, 800 features, labels +1 / −1):

| File | Role | Examples |
|---|---|---|
| `first_test_3_1.txt` | Used to train the model | 21,017 |
| `first_train_3_1.txt` | Predicted by the model | 84,068 |

## Model
Linear Support Vector Machine (`LinearSVC`, `C = 1.0`, scikit-learn).

## How to Run
```bash
mkdir -p data && bsdtar -xf SVM.rar -C data
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python svm_svmlight.py --train data/first_test_3_1.txt --evaluate data/first_train_3_1.txt
```

## Result

### Accuracy: **84.13%** (70,728 of 84,068 correctly classified)

| | Predicted −1 | Predicted +1 |
|---|---|---|
| **Actual −1** | 50,735 | 4,999 |
| **Actual +1** | 8,341 | 19,993 |

**Summary:**
- The SVM learned from 21,017 examples and correctly predicted **84.13%** of the 84,068 unseen examples.
- Accuracy on its own training data was 85.56%, close to the evaluation result, so the model generalizes well and does not overfit.
- It beats the 66.3% baseline of always guessing the most common class (−1).
- Class −1 is predicted very well (91.0% found). Class +1 is harder (70.6% found), since it has fewer examples to learn from.
# DataMining-SVM
