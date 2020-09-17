import sys
sys.path.append("../")

import threading
import numpy as np
import matplotlib.pyplot as plt
from time import time
from snorkel.analysis import Scorer

from Our_Monitors.CD_Monitor import Informed_LabelModel, CDM
#from Our_Monitors.CDGA_Monitor import CDGAM
from Our_Monitors.CDGA_Monitor_fast import CDGAM_fast as CDGAM
#from Our_Monitors.New_Monitor import NM
from Our_Monitors.New_Monitor_fast import NM_fast as NM

import utils
import importlib
importlib.reload(utils)
from utils import predict_at_abstain_rate

np.random.seed(int(time()))

# Ensure data matricies are defined
try:
    L_train_local = np.copy(L_train)                                    # 80% train
    L_dev_local, Y_dev_local = np.copy(L_dev), np.copy(Y_dev)           # 10% dev
    L_test_local, Y_test_local = np.copy(L_test), np.copy(Y_test)       # 10% test
except:
    raise Exception("Data not defined properly...")

# Extract experiment parameters from arguments
n_exps = int(sys.argv[1])
n_iters = int(sys.argv[2])              # N
n_subsets = int(sys.argv[3])            # J
subset_size = int(sys.argv[4])          # K
with_replacement = bool(sys.argv[5])
monitor = str(sys.argv[6])
abstain_rate = float(sys.argv[7])       # if < 0 then no abstain rate requested

# Other parameters
sig = 0.05
policy = "old"
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

def find_deps(m, L_dev, Y_dev):
    if m == "CDM":
        return CDM(L_dev, Y_dev, k=2, sig=sig, policy=policy, verbose=False, return_more_info=False)
    elif m == "CDGAM":
        return CDGAM(L_dev, k=2, sig=sig, policy=policy, verbose=False, return_more_info=False)
    elif m == "NM":
        return NM(L_dev, Y_dev, k=2, sig=sig, policy=policy, verbose=False, return_more_info=False)
    else:
        raise Exception("Invalid monitor...")

def thread_experiment(exp, L_train, L_dev, Y_dev, L_test, Y_test):
    for iter in range(n_iters):
        # Randomly sample J sets of K LFs
        subsets = np.random.choice(L_train.shape[1], size=(n_subsets,subset_size), replace=with_replacement)

        # Define a new LF for each of the J sets as the prediction of a dependency-informed 
        # Snorkel model with the K LFs    
        new_L_train = np.zeros((L_train.shape[0],n_subsets))
        new_L_dev = np.zeros((L_dev.shape[0],n_subsets))
        new_L_test = np.zeros((L_test.shape[0],n_subsets))  

        for i, subset in enumerate(subsets):
            print(subset)
            deps = find_deps(monitor, L_dev[:,subset], Y_dev)
            print(len(deps))
            il_model = Informed_LabelModel(deps, cardinality=2)
            il_model.fit(L_train[:,subset], n_epochs=n_epochs, lr=lr)

            new_L_train[:,i] = il_model.predict(L_train[:,subset])
            new_L_dev[:,i] = il_model.predict(L_dev[:,subset])
            new_L_test[:,i] = il_model.predict(L_test[:,subset])
            
        L_train = np.copy(new_L_train)
        L_dev = np.copy(new_L_dev)
        L_test = np.copy(new_L_test)

        # Train and evaluate a dependency-informed Snorkel model
        try:
            deps = find_deps(monitor, L_dev, Y_dev)
            il_model = Informed_LabelModel(deps, cardinality=2)
            il_model.fit(L_train, n_epochs=n_epochs, lr=lr)

            if abstain_rate < 0:
                Y_pred = il_model.predict(L_test)
            else:
                Y_prob = il_model.predict_proba(L_test)
                Y_pred = predict_at_abstain_rate(Y_prob, abstain_rate)

            iter_scores = scorer.score(Y_test, preds=Y_pred)
            iter_scores["num deps"] = len(deps)
            print(iter_scores)
            
            mutex.acquire()
            results_mtx[exp,iter,:] = list(iter_scores.values())
            mutex.release()
        except Exception as e:
            print("Thread {}: Iter {}: {}".format(exp,iter,e))
            continue


## Run the experiment several times
threads = []

for exp in range(n_exps):
    print("Main: create and start experiment thread {}".format(exp)) 
    x = threading.Thread(target=thread_experiment, args=(exp, L_train_local, L_dev_local, Y_dev_local, L_test_local, Y_test_local))
    threads.append(x)
    x.start()

for exp, thread in enumerate(threads):
    thread.join()
    print("Main: experiment thread {} done".format(exp))


## Save results
np.save("e6_(" + ",".join(sys.argv[1:]) + ").npy", results_mtx)


## Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(4,10))
fig.suptitle("E6 Results (exps=" + str(n_exps)
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
plt.show()