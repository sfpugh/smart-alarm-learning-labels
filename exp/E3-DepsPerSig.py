import sys
sys.path.append("../")

import logging
import matplotlib.pyplot as plt
import numpy as np

from Our_Monitors.CD_Monitor import CDM
from Our_Monitors.CDGA_Monitor_fast import CDGAM_fast as CDGAM
from Our_Monitors.New_Monitor_fast import NM_fast as NM


# Copy data from source notebook
L_Dev = np.copy(L_dev)
Y_Dev = np.copy(Y_dev)

# Parameters
policy = "old"

# Find all possible dependencies, i.e., dont set a sig
CDM_deps = CDM(L_Dev, Y_Dev, k=2, sig=1.1, policy=policy, verbose=False, return_more_info=True)
logging.info("CDM finished...")
CDGAM_deps = CDGAM(L_Dev, k=2, sig=1.1, policy=policy, verbose=False, return_more_info=True)
logging.info("CDGAM finished...")
NM_deps = NM(L_Dev, Y_Dev, k=2, sig=1.1, policy=policy, verbose=False, return_more_info=True)
logging.info("NM finished...")

# Determine number of dependencies per sig level
sigs = np.linspace(0, 1, num=100, dtype=float)
results = np.zeros((3, len(sigs)), dtype=int)

for i, sig in enumerate(sigs):
        results[0,i] = len([x for x in zip(CDM_deps["CD_edges"], CDM_deps["CD_edges_p_vals"]) if x[1] < sig])
        results[1,i] = len([x for x in zip(CDGAM_deps["CD_edges"], CDGAM_deps["CD_edges_p_vals"]) if x[1] < sig])
        results[2,i] = len([x for x in zip(NM_deps["CD_edges"], NM_deps["CD_edges_p_vals"]) if x[1] < sig])

# Plot
fig, ax = plt.subplots()
ax.plot(sigs, results[0,:], label="CDM")
ax.plot(sigs, results[1,:], label="CDGAM")
ax.plot(sigs, results[2,:], label="NM")
ax.legend(bbox_to_anchor=(1.3,1))
ax.set(xlabel="sig", 
        ylabel="number of dependencies", 
        title="Number of Dependencies for Increasing Significance Level (" + EXAMPLE + ")")
plt.show()