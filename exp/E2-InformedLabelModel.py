import sys
sys.path.append("../")
import logging
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from Our_Monitors.CD_Monitor import Informed_LabelModel
from Our_Monitors.CDGA_Monitor_fast import CDGAM_fast as CDGAM
from snorkel.analysis import Scorer

import utils
import importlib
importlib.reload(utils)
from utils import predict_at_abstain_rate


# Extract parameters from arguments
sig = float(sys.argv[1])
n_epochs = int(sys.argv[2])
lr = float(sys.argv[3])
l2 = float(sys.argv[4])
optimizer = str(sys.argv[5])
lr_scheduler = str(sys.argv[6])
abstain_rate = float(sys.argv[7])   # if < 0 then no abstain rate requested

# Other parameters
n_folds = 5
policy = "old"

# Extract relevant data
L_data_local = np.copy(L_data)
Y_data_local = np.copy(Y_data)

# Set up Scorer
my_metrics = {"abstain rate": lambda golds, preds, probs: np.sum(preds == ABSTAIN) / len(preds)}
scorer = Scorer(metrics=["accuracy","f1"], custom_metric_funcs=my_metrics)

# Cross validation
all_scores = []
kf = KFold(n_splits=n_folds, shuffle=True)

for i, (train_dev_idx, test_idx) in enumerate(kf.split(L_data_local)):
    #train_idx, dev_idx = train_test_split(train_dev_idx, test_size=0.25, shuffle=True)
    # Define train dataset
    L_train = L_data_local[train_dev_idx]
    Y_train = Y_data_local[train_dev_idx]
    # Define development dataset
    L_dev = L_data_local[train_dev_idx]
    Y_dev = Y_data_local[train_dev_idx]
    # Define test dataset
    L_test = L_data_local[test_idx]
    Y_test = Y_data_local[test_idx]

    # Learn dependencies
    deps = CDGAM(L_dev, k=2, sig=sig, policy=policy, verbose=False, return_more_info=False)

    # Evaluate a dependency-informed Snorkel model
    il_model = Informed_LabelModel(deps, cardinality=2, verbose=False)
    il_model.fit(L_train, n_epochs=n_epochs, lr=lr, l2=l2, optimizer=optimizer, lr_scheduler=lr_scheduler)

    try:
        if abstain_rate < 0:
            Y_pred = il_model.predict(L_test, tie_break_policy="abstain")
        else:
            Y_prob = il_model.predict_proba(L_test)
            Y_pred = predict_at_abstain_rate(Y_prob, abstain_rate)

        scores = scorer.score(Y_test, preds=Y_pred)
        scores["num deps"] = len(deps)
        all_scores.append(scores)
    except Exception as e:
        print("Iter {}: {}".format(i+1,e))
        continue
    
    # Logging
    logging.INFO("Iteration " + str(i+1) + ":" + str(scores))

print("-- SUMMARY --")
print("accuracy: AVG {:.3f}, STD {:.3f}".format(np.mean([s["accuracy"] for s in all_scores]), np.std([s["accuracy"] for s in all_scores])))
print("f1: AVG {:.3f}, STD {:.3f}".format(np.mean([s["f1"] for s in all_scores]), np.std([s["f1"] for s in all_scores])))
print("abstain rate: AVG {:.3f}, STD {:.3f}".format(np.mean([s["abstain rate"] for s in all_scores]), np.std([s["abstain rate"] for s in all_scores])))
print("num deps: AVG {:.3f}, STD {:.3f}".format(np.mean([s["num deps"] for s in all_scores]), np.std([s["num deps"] for s in all_scores])))