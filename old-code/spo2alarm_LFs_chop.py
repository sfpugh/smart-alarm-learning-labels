from snorkel.labeling import labeling_function
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

@labeling_function()
def lf_short_alarm_15s(x):
    """
    If the SpO2-Low alarm duration is at most 15 seconds then the alarm is suppressible, 
    otherwise abstain 
    """
    return SUPPRESSIBLE if x.duration <= 15 else ABSTAIN

@labeling_function()
def lf_short_alarm_10s(x):
    """
    If the SpO2-Low alarm duration is at most 10 seconds then the alarm is suppressible, 
    otherwise abstain 
    """
    return SUPPRESSIBLE if x.duration <= 10 else ABSTAIN

@labeling_function()
def lf_short_alarm_5s(x):
    """
    If the SpO2-Low alarm duration is at most 5 seconds then the alarm is suppressible, 
    otherwise abstain 
    """
    return SUPPRESSIBLE if x.duration <= 5 else ABSTAIN

@labeling_function()
def lf_immediate_recovery_10s(x):
    """
    If SpO2 level increases/recovers by more than 20 percentage points within 
    10 seconds of alarm start then the alarm is suppressible, otherwise abstain
    """
    return ABSTAIN 

@labeling_function()
def lf_immediate_recovery_15s(x):
    """
    If SpO2 level increases/recovers by more than 30 percentage points within 
    15 seconds of alarm start then the alarm is suppressible, otherwise abstain
    """
    return ABSTAIN

@labeling_function()
def lf_hr_tech_err_20(x):
    """
    If the difference between the SpO2 HR and ECG HR is larger than 20 percentage points 
    at time of alarm then suppressible, otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    t = v_df.index.get_loc(x.alarm_datetime, method='bfill')  # or use 'nearest' to get closest timestamp in general
    return SUPPRESSIBlE if abs(v_df.at[t,'SPO2-R'] - v_df.at[t,'HR']) > 20 else ABSTAIN
    #return SUPPRESSIBlE if abs(v_df['SPO2-R'].loc[t] - v_df['HR'].loc[t]) > 20 else ABSTAIN

@labeling_function()
def lf_hr_tech_err_30(x):
    """
    If the difference between the SpO2 HR and ECG HR is larger than 30 percentage points 
    at time of alarm then suppressible, otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    t = v_df.index.get_loc(x.alarm_datetime, method='bfill')  # or use 'nearest' to get closest timestamp in general
    return SUPPRESSIBlE if abs(v_df['SPO2-R'].loc[t] - v_df['HR'].loc[t]) > 30 else ABSTAIN

@labeling_function()
def lf_long_alarm_60s(x):
    """
    If the alarm duration is at least 60 seconds then the alarm is not suppressible, 
    otherwise abstain 
    """
    return NOT_SUPPRESSIBLE if x.duration >= 60 else ABSTAIN

@labeling_function()
def lf_long_alarm_65s(x):
    """
    If the alarm duration is at least 65 seconds then the alarm is not suppressible, 
    otherwise abstain 
    """
    return NOT_SUPPRESSIBLE if x.duration >= 65 else ABSTAIN

@labeling_function()
def lf_long_alarm_70s(x):
    """
    If the alarm duration is at least 70 seconds then the alarm is not suppressible, 
    otherwise abstain 
    """
    return NOT_SUPPRESSIBLE if x.duration >= 70 else ABSTAIN

@labeling_function()
def lf_spo2_below85_over120s(x):
    """
    If SpO2 level stays within range (80,85] for longer than 120 seconds since alarm start 
    then the alarm is not suppressible, otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    spo2_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=121), ['SPO2-%']]
    return NOT_SUPPRESSIBLE if np.all( (80 < spo2_subrange) & (spo2_subrange <= 85) ) else ABSTAIN

@labeling_function()
def lf_spo2_below80_over100s(x):
    """
    If SpO2 level stays within range (70,80] for longer than 100 seconds since alarm start 
    then the alarm is not suppressible, otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    spo2_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=101), ['SPO2-%']]
    return NOT_SUPPRESSIBLE if np.all( (70 < spo2_subrange) & (spo2_subrange <= 80) ) else ABSTAIN

@labeling_function()
def lf_spo2_below70_over90s(x):
    """
    If SpO2 level stays within range (60,70] for longer than 90 seconds since alarm start 
    then the alarm is not suppressible, otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    spo2_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=91), ['SPO2-%']]
    return NOT_SUPPRESSIBLE if np.all( (60 < spo2_subrange) & (spo2_subrange <= 70) ) else ABSTAIN

@labeling_function()
def lf_spo2_below60_over60s(x):
    """
    If SpO2 level stays within range (50,60] for longer than 60 seconds since alarm start 
    then the alarm is not suppressible, otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    spo2_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=61), ['SPO2-%']]
    return NOT_SUPPRESSIBLE if np.all( (50 < spo2_subrange) & (spo2_subrange <= 60) ) else ABSTAIN

@labeling_function()
def lf_spo2_below50_over30s(x):
    """
    If SpO2 level stays within range (0,50] for longer than 30 seconds since alarm start 
    then the alarm is not suppressible, otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    spo2_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=31), ['SPO2-%']]
    return NOT_SUPPRESSIBLE if np.all( spo2_subrange <= 50 ) else ABSTAIN

@labeling_function()
def lf_hr_below50_over120s(x):
    """
    If HR below 50 * age factor for over 120 seconds since alarm start then the alarm is not suppressible, 
    otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    hr_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=121), ['HR']]
    return NOT_SUPPRESSIBLE if np.all( (40*AGE_FACTOR < hr_subrange) & (hr_subrange <= 50*AGE_FACTOR) ) else ABSTAIN

@labeling_function()
def lf_hr_below40_over60s(x):
    """
    If HR below 40 * age factor for over 60 seconds since alarm start then the alarm is not suppressible, 
    otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    hr_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=61), ['HR']]
    return NOT_SUPPRESSIBLE if np.all( (30*AGE_FACTOR < hr_subrange) & (hr_subrange <= 40*AGE_FACTOR) ) else ABSTAIN

@labeling_function()
def lf_hr_below30(x):
    """
    If HR below 30 * age factor for any duration since alarm start then the alarm is not suppressible, 
    otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    hr_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=x.duration), ['HR']]
    return NOT_SUPPRESSIBLE if np.any( hr_subrange <= 30 * AGE_FACTOR ) else ABSTAIN

@labeling_function()
def lf_rr_below50_over120s(x):
    """
    If HR below 50 * age factor for over 120 seconds since alarm start then the alarm is not suppressible, 
    otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    hr_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=121), ['HR']]
    return NOT_SUPPRESSIBLE if np.all( (40*AGE_FACTOR < hr_subrange) & (hr_subrange <= 50*AGE_FACTOR) ) else ABSTAIN

@labeling_function()
def lf_rr_below40_over60s(x):
    """
    If RR below 40 * age factor for over 60 seconds since alarm start then the alarm is not suppressible, 
    otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    hr_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=61), ['RESP']]
    return NOT_SUPPRESSIBLE if np.all( (30*AGE_FACTOR < hr_subrange) & (hr_subrange <= 40*AGE_FACTOR) ) else ABSTAIN

@labeling_function()
def lf_rr_below30(x):
    """
    If RR below 30 * age factor for any duration since alarm start then the alarm is not suppressible, 
    otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    hr_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=x.duration), ['RESP']]
    return NOT_SUPPRESSIBLE if np.any( rr_subrange <= 30 * AGE_FACTOR ) else ABSTAIN

@labeling_function()
def lf_repeat_alarms_within30s(x):
    """
    If there is more than one SpO2 alarm within 30 seconds of the current alarm start time then 
    the alarm is not suppressible, otherwise abstain
    """
    temp = alarms_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=30), :]
    temp = temp[ temp['pt_id'] == x.pt_id ]
    return NOT_SUPPRESSIBLE if temp.shape[0] > 1 else ABSTAIN

@labeling_function()
def lf_repeat_alarms_within60s(x):
    """
    If there is more than one SpO2 alarm within 60 seconds of the current alarm start time then 
    the alarm is not suppressible, otherwise abstain
    """
    temp = alarms_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=60), :]
    temp = temp[ temp['pt_id'] == x.pt_id ]
    return NOT_SUPPRESSIBLE if temp.shape[0] > 1 else ABSTAIN

@labeling_function()
def lf_repeat_alarms_within90s(x):
    """
    If there is more than one SpO2 alarm within 90 seconds of the current alarm start time then 
    the alarm is not suppressible, otherwise abstain
    """
    temp = alarms_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=90), :]
    temp = temp[ temp['pt_id'] == x.pt_id ]
    return NOT_SUPPRESSIBLE if temp.shape[0] > 1 else ABSTAIN
