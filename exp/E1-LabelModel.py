import sys
import logging
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from snorkel.labeling.model import LabelModel
from snorkel.analysis import Scorer

import utils
import importlib
importlib.reload(utils)
from utils import predict_at_abstain_rate


# Extract parameters from arguments
n_epochs = int(sys.argv[1])
lr = float(sys.argv[2])
l2 = float(sys.argv[3])
optimizer = str(sys.argv[4])
lr_scheduler = str(sys.argv[5])
abstain_rate = float(sys.argv[6])   # if < 0 then no abstain rate requested

# Other parameters
n_folds = 5

# Extract relevant data
L_data_local = np.copy(L_data[:,:57])
Y_data_local = np.copy(Y_data)

# Set up Scorer
my_metrics = {"abstain rate": lambda golds, preds, probs: np.sum(preds == ABSTAIN) / len(preds)}
scorer = Scorer(metrics=["accuracy","f1"], custom_metric_funcs=my_metrics)

# Cross validation
all_scores = []
kf = KFold(n_splits=n_folds, shuffle=True)

for i, (train_idx, test_idx) in enumerate(kf.split(L_data_local)):
    # Define train dataset
    L_train = L_data_local[train_idx]
    Y_train = Y_data_local[train_idx]
    # Define test dataset
    L_test = L_data_local[test_idx]
    Y_test = Y_data_local[test_idx]

    # Evaluate a dependency-informed Snorkel model
    l_model = LabelModel(cardinality=2, verbose=False)
    l_model.fit(L_train, n_epochs=n_epochs, lr=lr, l2=l2, optimizer=optimizer, lr_scheduler=lr_scheduler)

    try:
        if abstain_rate < 0:
            Y_pred = l_model.predict(L_test, tie_break_policy="abstain")
        else:
            Y_prob = l_model.predict_proba(L_test)
            Y_pred = predict_at_abstain_rate(Y_prob, abstain_rate)

        scores = scorer.score(Y_test, preds=Y_pred)
        all_scores.append(scores)
    except Exception as e:
        print("Iter {}: {}".format(i+1,e))
        continue
    
    # Logging
    print("Iteration " + str(i+1) + ":", scores)

print("-- SUMMARY --")
print("accuracy: AVG {:.3f}, STD {:.3f}".format(np.mean([s["accuracy"] for s in all_scores]), np.std([s["accuracy"] for s in all_scores])))
print("f1: AVG {:.3f}, STD {:.3f}".format(np.mean([s["f1"] for s in all_scores]), np.std([s["f1"] for s in all_scores])))
print("abstain rate: AVG {:.3f}, STD {:.3f}".format(np.mean([s["abstain rate"] for s in all_scores]), np.std([s["abstain rate"] for s in all_scores])))