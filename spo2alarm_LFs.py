from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.preprocess import preprocessor
import pandas as pd
import numpy as np
import wfdb

MIMIC3WDB = 'mimic3wdb/matched/'

ABSTAIN = -1
NOT_SUPPRESSIBLE = 0
SUPPRESSIBLE = 1

@preprocessor()
def retrieve_spo2_waveform(x):
    rec_path = x.mimic3wdb_matched_ref.split('/')
    pb_dir = MIMIC3WDB + rec_path[0] + '/' + rec_path[1]

    try:
        record = wfdb.rdrecord(rec_path[2], channel_names=['SpO2'], pb_dir=pb_dir)
    except:
        record = wfdb.rdrecord(rec_path[2], channel_names=['%SpO2'], pb_dir=pb_dir)

    x.spo2 = record.p_signal[:,0]
    
    return x

@preprocessor()
def retrieve_hr_waveform(x):
    rec_path = x.mimic3wdb_matched_ref.split('/')
    pb_dir = MIMIC3WDB + rec_path[0] + '/' + rec_path[1]

    record = wfdb.rdrecord(rec_path[2], channel_names=['HR'], pb_dir=pb_dir)
    x.hr = record.p_signal[:,0]
    
    return x

@preprocessor()
def retrieve_rr_waveform(x):
    rec_path = x.mimic3wdb_matched_ref.split('/')
    pb_dir = MIMIC3WDB + rec_path[0] + '/' + rec_path[1]

    record = wfdb.rdrecord(rec_path[2], channel_names=['RESP'], pb_dir=pb_dir)
    x.resp = record.p_signal[:,0]
    
    return x

## Suppressible guidelines
@labeling_function()
def lf_alarm_too_short(x):
    return SUPPRESSIBLE if x.alarm_end - x.alarm_start + 1 <= 15  else ABSTAIN

@labeling_function(pre=[retrieve_spo2_waveform])
def lf_immediate_recovery_10s(x):
    return SUPPRESSIBLE if abs(x.spo2[x.alarm_start] - x.spo2[x.alarm_start+10]) > 20 else ABSTAIN

@labeling_function(pre=[retrieve_spo2_waveform])
def lf_immediate_recovery_15s(x):
    return SUPPRESSIBLE if abs(x.spo2[x.alarm_start] - x.spo2[x.alarm_start+15]) > 30 else ABSTAIN

## Not-suppressible guidelines
@labeling_function(pre=[retrieve_spo2_waveform])
def lf_spo2_below85_over120s(x):
    return NOT_SUPPRESSIBLE if np.all( (80 < x.spo2[x.alarm_start:x.alarm_start+120+1]) & (x.spo2[x.alarm_start:x.alarm_start+120+1] <= 85) ) else ABSTAIN

@labeling_function(pre=[retrieve_spo2_waveform])
def lf_spo2_below80_over100s(x):
    return NOT_SUPPRESSIBLE if np.all( (70 < x.spo2[x.alarm_start:x.alarm_start+100+1]) & (x.spo2[x.alarm_start:x.alarm_start+100+1] <= 80) ) else ABSTAIN

@labeling_function(pre=[retrieve_spo2_waveform])
def lf_spo2_below70_over90s(x):
    return NOT_SUPPRESSIBLE if np.all( (60 < x.spo2[x.alarm_start:x.alarm_start+90+1]) & (x.spo2[x.alarm_start:x.alarm_start+90+1] <= 70) ) else ABSTAIN

@labeling_function(pre=[retrieve_spo2_waveform])
def lf_spo2_below60_over60s(x):
    return NOT_SUPPRESSIBLE if np.all( (50 < x.spo2[x.alarm_start:x.alarm_start+60+1]) & (x.spo2[x.alarm_start:x.alarm_start+60+1] <= 60) ) else ABSTAIN

@labeling_function(pre=[retrieve_spo2_waveform])
def lf_spo2_below50_over30s(x):
    return NOT_SUPPRESSIBLE if np.all( x.spo2[x.alarm_start:x.alarm_start+30+1] <= 50 ) else ABSTAIN

@labeling_function()
def lf_alarm_too_long(x):
    return NOT_SUPPRESSIBLE if x.alarm_end - x.alarm_start >= 60 else ABSTAIN

@labeling_function(pre=[retrieve_hr_waveform])
def lf_hr_below50_over120s(x):
    return ABSTAIN

@labeling_function(pre=[retrieve_hr_waveform])
def lf_hr_below40_over60s(x):
    return ABSTAIN

@labeling_function(pre=[retrieve_hr_waveform])
def lf_hr_below30(x):
    return ABSTAIN

@labeling_function(pre=[retrieve_rr_waveform])
def lf_rr_below50_over120s(x):
    return ABSTAIN

@labeling_function(pre=[retrieve_rr_waveform])
def lf_rr_below40_over60s(x):
    return ABSTAIN

@labeling_function(pre=[retrieve_rr_waveform])
def lf_rr_below30(x):
    return ABSTAIN
