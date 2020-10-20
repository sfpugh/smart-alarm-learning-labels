import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from time import time
from itertools import combinations

from snorkel.analysis import Scorer
from utils import predict_at_abstain_rate
from Our_Monitors.CD_Monitor import Informed_LabelModel

# Extract experimental parameters from arguments
n_exps = int(sys.argv[1])
n_iters = int(sys.argv[2])              # N
n_subsets = int(sys.argv[3])            # J
subset_size = int(sys.argv[4])          # K
with_replacement = bool(sys.argv[5])
abstain_rate = float(sys.argv[6])       # if < 0 then no abstain rate requested

# Run source notebook and to set up data, labeling functions, and label matricies
#from canned_example.canned_STABLE import *
sys.path.append("/home/sfpugh/smart-alarm-learning-labels/smart_alarm")
from smart_alarm.learn_labels_STABLE import *
try:
    L_data
    Y_data
except:
    print("No L_data and/or Y_data found in source notebook.")
    sys.exit(1)

# Snorkel model parameters
n_epochs = 100
lr = 0.01

# Set up Scorer
n_metrics = 3
my_metrics = {"abstain rate": lambda golds, preds, probs: np.sum(preds == ABSTAIN) / len(preds)}
scorer = Scorer(metrics=["accuracy","f1"], custom_metric_funcs=my_metrics)


## Define the experiment
def experiment(exp, L_Data, Y_Data):
    # Seed numpy to get different sequences of subsets
    np.random.seed(exp)

    i = 0
    while i < n_iters:
        i += 1
        
        # Randomly sample J sets of K LFs
        subsets = np.random.choice(L_Data.shape[1], size=(n_subsets,subset_size), replace=with_replacement)
        print(subsets)
        return 0
        
        # Define a new LF for each of the J sets as the prediction of a dependency-informed Snorkel model with the K LFs    
        new_L_Data = np.zeros((L_Data.shape[0],n_subsets))

        for j, subset in enumerate(subsets):
            deps = list( combinations(range(subset_size), 2) )    # assume LF dependency graph is fully connected

            il_model = Informed_LabelModel(deps, cardinality=2)
            il_model.fit(L_Data[:,subset], n_epochs=n_epochs, lr=lr)

            new_L_Data[:,j] = il_model.predict(L_Data[:,subset])
            
        L_Data = np.copy(new_L_Data)        
    
    # Randomly sample K LFs
    subset = np.random.choice(L_Data.shape[1], size=subset_size, replace=with_replacement)
    
    # Evaluate the last K LFs
    deps = list( combinations(range(subset_size), 2) )    # assume LF dependency graph is fully connected
    
    il_model = Informed_LabelModel(deps, cardinality=2)
    il_model.fit(L_Data[:,subset], n_epochs=n_epochs, lr=lr)
    
    Y_pred = il_model.predict(L_Data[:,subset])
    scores = scorer.score(Y_Data, preds=Y_pred)
    print("Experiment {} scores: {}".format(exp, scores))



## Run the experiment several times
processes = []

for exp in range(n_exps):
    print("Main: create and start experiment {}".format(exp+1)) 
    x = multiprocessing.Process(target=experiment, args=(exp+1, L_data, Y_data))
    processes.append(x)
    x.start()

for exp, process in enumerate(processes):
    process.join()
    print("Main: experiment {} done".format(exp+1))


#for exp in range(n_exps):
#    print("Experiment {} of {} running...".format(exp+1, n_exps))
#    experiment(exp, L_data, Y_data)
