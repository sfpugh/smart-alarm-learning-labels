#!/usr/bin/env python
# coding: utf-8

# # Canned Example: sign(x)
# This notebook defines our canned example which defines several labeling functions that model the sign function. (See Overleaf document for full definition of this example.)

# In[1]:


import numpy as np
from snorkel.labeling import labeling_function, LFApplier, LFAnalysis
from snorkel.labeling.model.label_model import LabelModel


# In[2]:


# Parameters for the analysis
N_LFS = 50
N_DPS = 10000

SEED_RNG = True
SEED = 1234


# In[3]:


# Defining labels for sake of clarity
ABSTAIN = -1
NEGATIVE = 0
POSITIVE = 1


# In[4]:


# Seed numpy.random for reproducible results
if SEED_RNG:
    np.random.seed(SEED)


# ## Define the Goal Model

# In[5]:


def f_star(x):
    return 1 if x == 0 else np.sign(x)


# ## Define the Dataset

# In[6]:


X_data = np.random.uniform(-1, 1, N_DPS)

f_star_vectorized = np.vectorize(f_star)
Y_data = f_star_vectorized(X_data)


# ## Define Labeling Functions

# In[7]:


def f(p_f, p_A, x):    
    # Draw from Bernoulli distribution
    z_f = np.random.binomial(1, p_f, 1)[0]
    z_A = np.random.binomial(1, p_A, 1)[0]
    
    x = int( (f_star(x) * z_f - f_star(x) * (1 - z_f)) * (1 - z_A) )
    if x == 1:
        return POSITIVE
    elif x == -1:
        return NEGATIVE
    else:
        return ABSTAIN


# In[18]:


# Generate several labeling functions for different values of p_f and p_A
lfs = []
for i in range(N_LFS):
    p_f = np.random.rand(1)[0]
    p_A = np.random.rand(1)[0]
    
    @labeling_function(name="LF{}({:.4f},{:.4f})".format(i, p_f, p_A))
    def lf(x):
        return f(p_f, p_A, x)
    
    lfs.append(lf)


# ## Apply Labeling Functions to the Data

# In[19]:


applier = LFApplier(lfs)
L_data = applier.apply(X_data, progress_bar=True)


# In[20]:


LFAnalysis(L_data, lfs=lfs).lf_summary(Y_data)


# ## Experiments
