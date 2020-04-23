from IPython.display import display
import matplotlib.pyplot as plt

import numpy as np
import os
import shutil
import wfdb

signals, fields = wfdb.rdsamp('mimic3wdb/31/3141595/3141595n', sampfrom=100, sampto=15000)
display(signals)
display(fields)