import sys
sys.path.append("../")

import logging
import matplotlib.pyplot as plt
import numpy as np

from Our_Monitors.CDGA_Monitor_fast import CDGAM_fast as CDGAM

print("running code from learn_labels.py...")
from smart_alarm.learn_labels import *

print("running E3...")

# Copy data from source notebook
try:
        L_dev = np.copy(L_data)
except e:
        raise Exception("Data not defined properly in source notebook/script...")

# Determine number of dependencies per sig level
sigs = np.linspace(0, 1, num=25, dtype=float)
results = np.zeros((2, len(sigs)), dtype=int)

for i, sig in enumerate(sigs):
        print("Running sig {}...".format(sig))

        deps = CDGAM(L_dev, k=2, sig=sig, policy="old", verbose=False, return_more_info=False)
        results[0,i] = len(deps)

        deps = CDGAM(L_dev, k=2, sig=sig, policy="new", verbose=False, return_more_info=False)
        results[1,i] = len(deps)

# save results
np.save("e3-results.npy", results)

# Plot
fig, ax = plt.subplots()
ax.plot(sigs, results[0,:], label="old")
ax.plot(sigs, results[1,:], label="new")
ax.legend()
ax.set(xlabel="sig", 
        ylabel="number of dependencies", 
        title="Number of Dependencies for Increasing Significance Level (CDGAM)")
plt.savefig('e3-results.png')
plt.show()
