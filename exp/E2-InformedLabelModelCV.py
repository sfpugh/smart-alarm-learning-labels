import sys
sys.path.append("../")

import numpy as np
from itertools import product
from snorkel.analysis import Scorer

from Our_Monitors.CD_Monitor import Informed_LabelModel, CDM
from Our_Monitors.CDGA_Monitor_fast import CDGAM_fast as CDGAM
from Our_Monitors.New_Monitor_fast import NM_fast as NM

import utils
import importlib
importlib.reload(utils)
from utils import predict_at_abstain_rate

# Extract arguments
monitor = str(sys.argv[1])
abstain_rate = float(sys.argv[2])

# Extract relevant data
L_Data = np.copy(L_data)
Y_Data = np.copy(Y_data)

# Set up Scorer
my_metrics = {"abstain rate": lambda golds, preds, probs: np.sum(preds == ABSTAIN) / len(preds)}
scorer = Scorer(metrics=["accuracy","f1"], custom_metric_funcs=my_metrics)

# Experiment
sig = np.flip( np.linspace(0, 0.5, num=50, dtype=float)[1:], 0 )
policy = ["new"]

n_epochs = [100, 500, 1000]
lr = [0.01, 0.05, 0.1]
l2 = [0.0, 0.01, 0.1]
optimizer = ["sgd", "adam", "adamax"]
lr_scheduler = ["constant", "linear", "exponential", "step"]

param_combos = list(product(sig, policy, n_epochs, lr, l2, optimizer, lr_scheduler))
results = np.empty((len(param_combos), 4), dtype=float)
results[:] = np.nan

prev_sig = None

for i, params in enumerate(param_combos):
    if prev_sig != params[0]:
        prev_sig = params[0]
        
        # Learn dependencies
        if monitor == "CDM":
            deps = CDM(L_Data, Y_data, k=2, sig=params[0], policy=params[1], verbose=False, return_more_info=False)
        elif monitor == "CDGAM":
            deps = CDGAM(L_Data, k=2, sig=params[0], policy=params[1], verbose=False, return_more_info=False)
        elif monitor == "NM":
            deps = NM(L_Data, Y_data, k=2, sig=params[0], policy=params[1], verbose=False, return_more_info=False)

    print(len(deps))
    results[i,3] = len(deps)
    
    # Train and evaluate an Informed Label Model
    try:
        il_model = Informed_LabelModel(deps, cardinality=2, verbose=False)
        il_model.fit(L_Data, n_epochs=params[2], lr=params[3], l2=params[4], optimizer=params[5], lr_scheduler=params[6])

        Y_prob = il_model.predict_proba(L_Data)
        Y_pred = predict_at_abstain_rate(Y_prob, abstain_rate)

        results[i,0:3] = list( scorer.score(Y_Data, preds=Y_pred).values() )
    except Exception as e:
        print("{}: {}".format(params, e))

np.save("e2-results-({},{}).npy".format(monitor, abstain_rate), results)