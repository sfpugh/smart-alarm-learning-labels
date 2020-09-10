import sys
from sklearn.model_selection import KFold, train_test_split
from Our_Monitors.CD_Monitor import Informed_LabelModel
from Our_Monitors.CDGA_Monitor_fast import CDGAM_fast as CDGAM
from snorkel.analysis import Scorer
from itertools import product

import my_utils
import importlib
importlib.reload(my_utils)
from my_utils import predict_at_abstain_rate

# Extract arguments
abstain_rate = float(sys.argv[1])

# Extract relevant data
L_data_local = np.copy(L_data[:,:57])
Y_data_local = np.copy(Y_data)

# Set up Scorer
my_metrics = {"abstain rate": lambda golds, preds, probs: np.sum(preds == ABSTAIN) / len(preds)}
scorer = Scorer(metrics=["accuracy","f1"], custom_metric_funcs=my_metrics)

# Parameters
n_folds = 5

sigs = [0.01, 0.05, 0.1]
policy = "new"
n_epochs = [100, 500, 1000]
lr = [0.01, 0.05, 0.1]
l2 = [0.0, 0.01, 0.1]
optimizer = ["sgd", "adam", "adamax"]
lr_scheduler = ["constant", "linear", "exponential", "step"]

all_params = list(product(sigs, n_epochs, lr, l2, optimizer, lr_scheduler))

# Cross validation
results = np.empty((len(all_params), n_folds, 3), dtype=float)
results[:] = np.nan

kf = KFold(n_splits=n_folds, shuffle=True)

for fold, (train_dev_idx, test_idx) in enumerate(kf.split(L_data_local)):
    train_idx, dev_idx = train_test_split(train_dev_idx, test_size=0.25, shuffle=True)
    # Define train dataset
    L_train, Y_train = L_data_local[train_idx], Y_data_local[train_idx]
    # Define development dataset
    L_dev, Y_dev = L_data_local[dev_idx], Y_data_local[dev_idx]
    # Define test dataset
    L_test, Y_test = L_data_local[test_idx], Y_data_local[test_idx]

    i = 0
    prev_sig = -1
    for params in all_params:
        # Learn dependencies
        if prev_sig != params[0]:
            deps = CDGAM(L_dev, k=2, sig=params[0], policy=policy, verbose=False, return_more_info=False)
            prev_sig = params[0]

        # Evaluate a dependency-informed Snorkel model
        il_model = Informed_LabelModel(deps, cardinality=2, verbose=False)
        il_model.fit(L_train, n_epochs=params[1], lr=params[2], l2=params[3], optimizer=params[4], lr_scheduler=params[5])

        try:
            if abstain_rate < 0:
                Y_pred = il_model.predict(L_test, tie_break_policy="abstain")
            else:
                Y_prob = il_model.predict_proba(L_test)
                Y_pred = predict_at_abstain_rate(Y_prob, abstain_rate)

            scores = scorer.score(Y_test, preds=Y_pred)
            results[i,fold,:] = list(scores.values())
        except Exception as e:
            print("Iter {}: {}".format(fold+1, e))
            continue

        i += 1

# Determine best parameters by average scores
cv_avg_scores = np.nanmean(results, axis=1)
i_best_acc = np.argmax(cv_avg_scores[:,0])
i_best_f1 = np.argmax(cv_avg_scores[:,1])

print("- Best Accuracy -")
print("params: ", all_params[i_best_acc])
print("scores: ", cv_avg_scores[i_best_acc])

print("- Best F1 -")
print("params: ", all_params[i_best_f1])
print("scores: ", cv_avg_scores[i_best_f1])