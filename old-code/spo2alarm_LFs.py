from snorkel.labeling import labeling_function
from snorkel.preprocess import preprocessor
import pandas as pd
import numpy as np
import wfdb

MIMIC3WDB = 'mimic3wdb/matched/'
ABSTAIN = -1
NOT_SUPPRESSIBLE = 0
SUPPRESSIBLE = 1

HR_AGE_FACTOR = 1
RR_AGE_FACTOR = 1

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


# If the alarm duration is less than or equal to 15 seconds then the alarm is suppressible, otherwise abstain
@labeling_function()
def lf_alarm_too_short(x):
    return SUPPRESSIBLE if x.alarm_end - x.alarm_start + 1 <= 15  else ABSTAIN


# If SpO2 level stays within range (70,85] for at most 60 seconds then the alarm is suppressible, otherwise abstain
@labeling_function(pre=[retrieve_spo2_waveform])
def lf_spo2_below85_atmost60s(x):
    return ABSTAIN


# If SpO2 level stays within range (60,70] for at most 45 seconds then the alarm is suppressible, otherwise abstain
@labeling_function(pre=[retrieve_spo2_waveform])
def lf_spo2_below70_atmost45s(x):
    return ABSTAIN


# If SpO2 level stays within range (50,60] for at most 20 seconds then the alarm is suppressible, otherwise abstain
@labeling_function(pre=[retrieve_spo2_waveform])
def lf_spo2_below60_atmost20s(x):
    return ABSTAIN


# If SpO2 level stays within range (0,50] for at most 15 seconds then the alarm is suppressible, otherwise abstain
@labeling_function(pre=[retrieve_spo2_waveform])
def lf_spo2_below50_atmost15s(x):
    return ABSTAIN


# If SpO2 level increases/recovers by more than 20 percentage points within 10 seconds of alarm start then the alarm is suppressible, otherwise abstain 
@labeling_function(pre=[retrieve_spo2_waveform])
def lf_immediate_recovery_10s(x):
    return SUPPRESSIBLE if (x.spo2[x.alarm_start+10] - x.spo2[x.alarm_start]) > 20 else ABSTAIN


# If SpO2 level increases/recovers by more than 30 percentage points within 15 seconds of alarm start then the alarm is suppressible, otherwise abstain 
@labeling_function(pre=[retrieve_spo2_waveform])
def lf_immediate_recovery_15s(x):
    return SUPPRESSIBLE if (x.spo2[x.alarm_start+15] - x.spo2[x.alarm_start]) > 30 else ABSTAIN


# If the alarm duration is greater than or equal to 60 seconds then the alarm is not suppressible, otherwise abstain
@labeling_function()
def lf_alarm_too_long(x):
    return NOT_SUPPRESSIBLE if x.alarm_end - x.alarm_start >= 60 else ABSTAIN


# If SpO2 level stays within range (80,85] for longer than 120 seconds since alarm start then the alarm is not suppressible, otherwise abstain
@labeling_function(pre=[retrieve_spo2_waveform])
def lf_spo2_below85_over120s(x):
    return NOT_SUPPRESSIBLE if np.all( (80 < x.spo2[x.alarm_start:x.alarm_start+120+1]) & (x.spo2[x.alarm_start:x.alarm_start+120+1] <= 85) ) else ABSTAIN


# If SpO2 level stays within range (70,80] for longer than 100 seconds since alarm start then the alarm is not suppressible, otherwise abstain
@labeling_function(pre=[retrieve_spo2_waveform])
def lf_spo2_below80_over100s(x):
    return NOT_SUPPRESSIBLE if np.all( (70 < x.spo2[x.alarm_start:x.alarm_start+100+1]) & (x.spo2[x.alarm_start:x.alarm_start+100+1] <= 80) ) else ABSTAIN


# If SpO2 level stays within range (60,70] for longer than 90 seconds since alarm start then the alarm is not suppressible, otherwise abstain
@labeling_function(pre=[retrieve_spo2_waveform])
def lf_spo2_below70_over90s(x):
    return NOT_SUPPRESSIBLE if np.all( (60 < x.spo2[x.alarm_start:x.alarm_start+90+1]) & (x.spo2[x.alarm_start:x.alarm_start+90+1] <= 70) ) else ABSTAIN


# If SpO2 level stays within range (50,60] for longer than 60 seconds since alarm start then the alarm is not suppressible, otherwise abstain
@labeling_function(pre=[retrieve_spo2_waveform])
def lf_spo2_below60_over60s(x):
    return NOT_SUPPRESSIBLE if np.all( (50 < x.spo2[x.alarm_start:x.alarm_start+60+1]) & (x.spo2[x.alarm_start:x.alarm_start+60+1] <= 60) ) else ABSTAIN


# If SpO2 level stays within range (0,50] for longer than 30 seconds since alarm start then the alarm is not suppressible, otherwise abstain
@labeling_function(pre=[retrieve_spo2_waveform])
def lf_spo2_below50_over30s(x):
    return NOT_SUPPRESSIBLE if np.all( x.spo2[x.alarm_start:x.alarm_start+30+1] <= 50 ) else ABSTAIN


# If HR below 50 * age factor for longer than 120 seconds since alarm start then the alarm is not suppressible, otherwise abstain
@labeling_function(pre=[retrieve_hr_waveform])
def lf_hr_below50_over120s(x):
    return NOT_SUPPRESSIBLE if np.all((40 * HR_AGE_FACTOR < x.hr[x.alarm_start:x.alarm_start+120+1]) & (x.hr[x.alarm_start:x.alarm_start+120+1] <= 50 * HR_AGE_FACTOR)) else ABSTAIN


# If HR below 40 * age factor for longer than 60 seconds since alarm start then the alarm is not suppressible, otherwise abstain
@labeling_function(pre=[retrieve_hr_waveform])
def lf_hr_below40_over60s(x):
    return NOT_SUPPRESSIBLE if np.all((30 * HR_AGE_FACTOR < x.hr[x.alarm_start:x.alarm_start+60+1]) & (x.hr[x.alarm_start:x.alarm_start+60+1] <= 40 * HR_AGE_FACTOR)) else ABSTAIN


# If HR below 30 * age factor for any duration since alarm start then the alarm is not suppressible, otherwise abstain
@labeling_function(pre=[retrieve_hr_waveform])
def lf_hr_below30(x):
    return NOT_SUPPRESSIBLE if np.any(x.hr[x.alarm_start:x.alarm_end] <= 30 * HR_AGE_FACTOR) else ABSTAIN


# If RR below 50 * age factor for longer than 120 seconds since alarm start then the alarm is not suppressible, otherwise abstain
@labeling_function(pre=[retrieve_rr_waveform])
def lf_rr_below50_over120s(x):
    return NOT_SUPPRESSIBLE if np.all((40 * RR_AGE_FACTOR < x.resp[x.alarm_start:x.alarm_start+120+1]) & (x.resp[x.alarm_start:x.alarm_start+120+1] <= 50 * RR_AGE_FACTOR)) else ABSTAIN


# If RR below 40 * age factor for longer than 60 seconds since alarm start then the alarm is not suppressible, otherwise abstain
@labeling_function(pre=[retrieve_rr_waveform])
def lf_rr_below40_over60s(x):
    return NOT_SUPPRESSIBLE if np.all((30 * RR_AGE_FACTOR < x.resp[x.alarm_start:x.alarm_start+60+1]) & (x.resp[x.alarm_start:x.alarm_start+60+1] <= 40 * RR_AGE_FACTOR)) else ABSTAIN


# If RR below 30 * age factor for any duration since alarm start then the alarm is not suppressible, otherwise abstain
@labeling_function(pre=[retrieve_rr_waveform])
def lf_rr_below30(x):
    return NOT_SUPPRESSIBLE if np.any(x.resp[x.alarm_start:x.alarm_end] <= 30 * RR_AGE_FACTOR) else ABSTAIN


