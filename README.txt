DATA MINING - SVM ASSIGNMENT
============================

TASK
----
Train an SVM on the "testing" dataset, use the model to predict the
"Training" dataset, and report the accuracy.


DATA
----
From SVM.rar (SVMlight format, 800 features, labels +1 / -1):

  first_test_3_1.txt   - used to train the model   - 21,017 examples
  first_train_3_1.txt  - predicted by the model    - 84,068 examples


MODEL
-----
Linear Support Vector Machine (LinearSVC, C = 1.0, scikit-learn).


HOW TO RUN
----------
  mkdir -p data && bsdtar -xf SVM.rar -C data
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  python svm_svmlight.py --train data/first_test_3_1.txt --evaluate data/first_train_3_1.txt


RESULT
------
ACCURACY: 84.13%  (70,728 of 84,068 correctly classified)

Confusion matrix:
                   Predicted -1    Predicted +1
  Actual -1           50,735           4,999
  Actual +1            8,341          19,993


SUMMARY
-------
- The SVM learned from 21,017 examples and correctly predicted 84.13% of
  the 84,068 unseen examples.
- Accuracy on its own training data was 85.56%, close to the evaluation
  result, so the model generalizes well and does not overfit.
- It beats the 66.3% baseline of always guessing the most common class (-1).
- Class -1 is predicted very well (91.0% found). Class +1 is harder
  (70.6% found), since it has fewer examples to learn from.
