from snorkel.labeling import LFAnalysis
from snorkel.labeling.apply.dask import PandasParallelLFApplier
from snorkel.labeling.model.label_model import LabelModel
from spo2alarm_LFs import *
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None   # to suppress runtime warnings for comparisons to NaN
np.warnings.filterwarnings('ignore')        # to suppress warnings of setting pandas column with copy

ABSTAIN = -1
NOT_SUPPRESSIBLE = 0
SUPPRESSIBLE = 1

## Read in spo2 alarm data
mimiciii_spo2alarms = pd.read_pickle('./spo2_alarm_data')
print('Read in ' + str(mimiciii_spo2alarms.shape[0]) + ' total alarms from the data set\n')

## Define training data
df_train = mimiciii_spo2alarms
#df_train = mimiciii_spo2alarms.head(100)

## Define set of LFs
lfs = [lf_alarm_too_short, lf_spo2_below85_over120s, lf_spo2_below80_over100s, lf_spo2_below70_over90s, lf_spo2_below60_over60s, lf_spo2_below50_over30s, lf_alarm_too_long]

## Apply the LFs to the unlabeled training data
print('Applying LFs to unlabeled training data...')
applier = PandasParallelLFApplier(lfs)
L_train = applier.apply(df=df_train, n_parallel=30, fault_tolerant=True)
print(LFAnalysis(L_train, lfs).lf_summary())

## Train the label model
print('Training the label model...')
label_model = LabelModel(cardinality=2)
label_model.fit(L_train)

## Analyze the LabelModel
print('Analyzing the LabelModel...')
df_train['label'] = label_model.predict(L=L_train, tie_break_policy="abstain")
#print(df_train.head(100))

print('# suppressible (1): ' + str(df_train[df_train.label == SUPPRESSIBLE].shape[0]))
print('# not suppressible (0): ' + str(df_train[df_train.label == NOT_SUPPRESSIBLE].shape[0]))
print('# abstain (-1): ' + str(df_train[df_train.label == ABSTAIN].shape[0]))

## Save the label model
label_model.save('./label_model.pkl')
