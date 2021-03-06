import sys
import logging
import threading
import numpy as np
from sklearn.model_selection import train_test_split
from Our_Monitors.CD_Monitor import Informed_LabelModel
#from Our_Monitors.CDGA_Monitor import CDGAM
from Our_Monitors.CDGA_Monitor_fast import CDGAM_fast as CDGAM
from snorkel.analysis import Scorer

import my_utils
import importlib
importlib.reload(my_utils)
from my_utils import predict_at_abstain_rate

import matplotlib.pyplot as plt


# Extract experiment parameters from arguments
n_exps = int(sys.argv[1])
n_iters = int(sys.argv[2])              # N
n_subsets = int(sys.argv[3])            # J
subset_size = int(sys.argv[4])          # K
with_replacement = bool(sys.argv[5])
abstain_rate = float(sys.argv[6])       # if < 0 then no abstain rate requested

# Other parameters
n_epochs = 100
lr = 0.01
sig = 0.05
policy = "new"

# Copy data from notebook
L_data_global = np.copy(L_alarms[:,:57])
Y_data_global = np.copy(Y_alarms)

# Set up Scorer
my_metrics = {"abstain rate": lambda golds, preds, probs: np.sum(preds == ABSTAIN) / len(preds)}
scorer = Scorer(metrics=["accuracy","f1"], custom_metric_funcs=my_metrics)

# Define the experiment
results_mtx = np.empty((n_exps,4,n_iters), dtype=float)
results_mtx[:] = np.nan

def thread_experiment(exp, L_data, Y_data):
    for iter in range(n_iters):
        # Randomly sample J sets of K LFs
        subsets = np.random.choice(L_data.shape[1], size=(n_subsets,subset_size), replace=with_replacement)

        # Define a new LF for each of the J sets as the prediction of a dependency-informed Snorkel model with the K LFs
        L_train, L_dev = train_test_split(L_data, test_size=0.2, shuffle=True)

        new_L_data = np.zeros((L_data.shape[0],n_subsets))         

        for i, subset in enumerate(subsets):
            deps = CDGAM(L_dev[:,subset], k=2, sig=sig, policy=policy, verbose=False, return_more_info=False)
            il_model = Informed_LabelModel(deps, cardinality=2)
            il_model.fit(L_train[:,subset], n_epochs=n_epochs, lr=lr)
            new_L_data[:,i] = il_model.predict(L_data[:,subset])
            
        L_data = np.copy(new_L_data)

        # Train a dependency-informed Snorkel model
        L_train, L_dev, Y_train, _ = train_test_split(L_data, Y_data, test_size=0.2, shuffle=True)

        deps = CDGAM(L_dev, k=2, sig=sig, policy=policy, verbose=False, return_more_info=False)
        il_model = Informed_LabelModel(deps, cardinality=2)
        il_model.fit(L_train, n_epochs=n_epochs, lr=lr)            

        # Evaluate model at specified abstain rate
        try:
            if abstain_rate < 0:
                Y_pred = il_model.predict(L_train)
            else:
                Y_prob = il_model.predict_proba(L_train)
                Y_pred = predict_at_abstain_rate(Y_prob, abstain_rate)

            iter_scores = scorer.score(Y_train, preds=Y_pred)
            iter_scores["num deps"] = len(deps)
            results_mtx[exp,:,iter] = list(iter_scores.values())
        except Exception as e:
            print("Thread {}: Iter {}: {}".format(exp,iter,e))
            continue


# Run the experiment several times
threads = []
for exp in range(n_exps):
    print("Main: create and start experiment thread {}".format(exp)) 
    x = threading.Thread(target=thread_experiment, args=(exp, L_data_global, Y_data_global))
    threads.append(x)
    x.start()

for exp, thread in enumerate(threads):
    thread.join()
    print("Main: experiment thread {} done".format(exp))


# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(4,10))
fig.suptitle("E6 results (exps=" + str(n_exps)
                        + ", N=" + str(n_iters) 
                        + ", J=" + str(n_subsets) 
                        + ", K=" + str(subset_size) 
                        + ", \nreplacement=" + str(with_replacement) 
                        + ", abstain=" + str(abstain_rate) 
                        + ")", y=0.95) 
# Accuracy
ax1.boxplot(results_mtx[:,0,::1], positions=range(1,n_iters+1))
ax1.set(ylabel="Accuracy")
ax1.set_ylim([0.0,1.0])
# F1
ax2.boxplot(results_mtx[:,1,::1], positions=range(1,n_iters+1))
ax2.set(ylabel="F1")
ax2.set_ylim([0.0,1.0])
# Num Dependencies
ax3.boxplot(results_mtx[:,3,::1], positions=range(1,n_iters+1))
ax3.set(xlabel="Iteration", ylabel="Num Dependencies (CDGAM)")
plt.show()

print("abstain rate: AVG {:.3f}, STD {:.3f} ".format(np.nanmean(results_mtx[:,2,:]), np.nanstd(results_mtx[:,2,:]))) 