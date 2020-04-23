from snorkel.labeling import PandasLFApplier, LFAnalysis
from snorkel.labeling.model.label_model import LabelModel
from spo2alarm_LFs import *
import pandas as pd
import numpy as np

ABSTAIN = -1
NOT_SUPPRESSIBLE = 0
SUPPRESSIBLE = 1

## Read in spo2 alarm data
mimiciii_spo2alarms = pd.read_pickle('./spo2_alarm_data')
print('There are ' + str(mimiciii_spo2alarms.shape[0]) + ' total alarms in the train data set\n')

## Define training data
df_train = mimiciii_spo2alarms.head(100)

## Define set of LFs
lfs = [lf_alarm_too_short, lf_spo2_below85_over120s, lf_spo2_below80_over100s, lf_spo2_below70_over90s, lf_spo2_below60_over60s, lf_spo2_below50_over30s, lf_alarm_too_long]

## Apply the LFs to the unlabeled training data
applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)
print(LFAnalysis(L_train, lfs).lf_summary())

## Train the label model
label_model = LabelModel(cardinality=2)
label_model.fit(L_train)
label_model.save('./label_model.pkl')

## Analysis of the training labels
df_train.loc[:,'label'] = label_model.predict(L=L_train, tie_break_policy="abstain")
df_train['label'] = label_model.predict(L=L_train, tie_break_policy="abstain")
print(df_train.head(100))

print('# suppressible (1): ' + str(df_train[df_train.label == SUPPRESSIBLE].shape[0]))
print('# not suppressible (0): ' + str(df_train[df_train.label == NOT_SUPPRESSIBLE].shape[0]))
print('# abstain (-1): ' + str(df_train[df_train.label == ABSTAIN].shape[0]))
