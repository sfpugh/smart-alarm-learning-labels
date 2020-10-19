import sys
sys.path.append("../")

import threading
import numpy as np
import matplotlib.pyplot as plt
from time import time
from snorkel.analysis import Scorer

from Our_Monitors.CD_Monitor import Informed_LabelModel
from Our_Monitors.CDGA_Monitor_fast import CDGAM_fast as CDGAM

import utils
import importlib
importlib.reload(utils)
from utils import predict_at_abstain_rate

# Seed numpy to get different sequences of subsets
np.random.seed(int(time()))

# Run source notebook
from canned_example.canned_STABLE import *

# Copy data from source notebook
#try:
#    L_data_local = np.copy(L_data)
#    Y_data_local = np.copy(Y_data)
#except:
#    raise Exception("Data not defined properly...")

# Extract experiment parameters from arguments
n_exps = int(sys.argv[1])
n_iters = int(sys.argv[2])              # N
n_subsets = int(sys.argv[3])            # J
subset_size = int(sys.argv[4])          # K
with_replacement = bool(sys.argv[5])
sig = float(sys.argv[6])
policy = str(sys.argv[7])
abstain_rate = float(sys.argv[8])       # if < 0 then no abstain rate requested

# Other parameters
n_epochs = 100
lr = 0.01

# Set up Scorer
n_metrics = 4
my_metrics = {"abstain rate": lambda golds, preds, probs: np.sum(preds == ABSTAIN) / len(preds)}
scorer = Scorer(metrics=["accuracy","f1"], custom_metric_funcs=my_metrics)


## Define the experiment
mutex = threading.Lock()
results_mtx = np.empty((n_exps,n_iters,n_metrics), dtype=float)
results_mtx[:] = np.nan

def thread_experiment(exp, L_data, Y_data):
    i = 0 
    while i < n_iters:
        # Randomly sample J sets of K LFs
        subsets = np.random.choice(L_data.shape[1], size=(n_subsets,subset_size), replace=with_replacement)

        # Define a new LF for each of the J sets as the prediction of a dependency-informed Snorkel model with the K LFs    
        new_L_data = np.zeros((L_data.shape[0],n_subsets))

        for j, subset in enumerate(subsets):
            deps = CDGAM(L_data[:,subset], k=2, sig=sig, policy=policy, verbose=False, return_more_info=False)

            il_model = Informed_LabelModel(deps, cardinality=2)
            il_model.fit(L_data[:,subset], n_epochs=n_epochs, lr=lr)

            new_L_data[:,j] = il_model.predict(L_data[:,subset])
            
        L_data = np.copy(new_L_data)

        # Train and evaluate a dependency-informed Snorkel model
        try:
            deps = CDGAM(L_data, k=2, sig=sig, policy=policy, verbose=False, return_more_info=False)
            print(len(deps))
            mutex.acquire()
            results_mtx[exp,iter,3] = len(deps)
            mutex.release()
            
            il_model = Informed_LabelModel(deps, cardinality=2)
            il_model.fit(L_data, n_epochs=n_epochs, lr=lr)

            if abstain_rate < 0:
                Y_pred = il_model.predict(L_data)
            else:
                Y_prob = il_model.predict_proba(L_data)
                Y_pred = predict_at_abstain_rate(Y_prob, abstain_rate)

            scores = scorer.score(Y_data, preds=Y_pred)
            
            mutex.acquire()
            results_mtx[exp,iter,0:3] = list(scores.values())
            mutex.release()
        except Exception as e:
            print("Thread {}: Iter {}: {}".format(exp,iter,e))
            continue


## Run the experiment several times
#threads = []

#for exp in range(n_exps):
#    print("Main: create and start experiment thread {}".format(exp)) 
#    x = threading.Thread(target=thread_experiment, args=(exp, L_data_local, Y_data_local))
#    threads.append(x)
#    x.start()

#for exp, thread in enumerate(threads):
#    thread.join()
#    print("Main: experiment thread {} done".format(exp))

for exp in range(n_exps):
    thread_experiment(exp, L_data, Y_data)
    print("exp {} finished...".format(exp))


## Save result matrix
np.save("e6_(" + ",".join(sys.argv[1:]) + ").npy", results_mtx)


## Plot
# We drop data for which we could not achieve the target abstain rate
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(4,10))
fig.suptitle("E6 Results (n_exps=" + str(n_exps)
                        + ", N=" + str(n_iters) 
                        + ", J=" + str(n_subsets) 
                        + ", K=" + str(subset_size) 
                        + ", \nreplacement=" + str(with_replacement) 
                        + ", abstain=" + str(abstain_rate) 
                        + ", sig=" + str(sig)
                        + ")", y=0.95) 
# Accuracy
mask = ~np.isnan(results_mtx[:,:,0])
d_acc = [m[i] for m, i in zip(results_mtx[:,:,0].T, mask.T)]
ax1.boxplot(d_acc, positions=range(1,n_iters+1))
ax1.set(ylabel="Accuracy")
ax1.set_ylim([0.0,1.0])
# F1
mask = ~np.isnan(results_mtx[:,:,1])
d_f1 = [m[i] for m, i in zip(results_mtx[:,:,1].T, mask.T)]
ax2.boxplot(d_f1, positions=range(1,n_iters+1))
ax2.set(ylabel="F1")
ax2.set_ylim([0.0,1.0])
# Num Dependencies
mask = ~np.isnan(results_mtx[:,:,3])
d_deps = [m[i] for m, i in zip(results_mtx[:,:,3].T, mask.T)]
ax3.boxplot(d_deps, positions=range(1,n_iters+1))
ax3.set(xlabel="Iteration", ylabel="Num Dependencies (CDGAM)")
# Save and display
plt.savefig("e6_(" + ",".join(sys.argv[1:]) + ").png")
plt.show()
