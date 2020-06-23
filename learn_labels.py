from snorkel.labeling.apply.dask import PandasParallelLFApplier
from snorkel.labeling.model.label_model import LabelModel
from snorkel.labeling import labeling_function, LFAnalysis
from sklearn.metrics import confusion_matrix
from read_chop_data import read_alarms, read_vitals
from datetime import datetime, timedelta
from matrixprofile import matrixProfile
import pandas as pd
import numpy as np


## Global vars
# Label mappings
ABSTAIN = -1
NOT_SUPPRESSIBLE = 0
SUPPRESSIBLE = 1

# Data dataframes
alarms_df = read_alarms()
vitals_dfs = read_vitals()
age_factors_df = pd.DataFrame({'pt_age_group':[1,2,3,4], 
                                    'pt_age_group_L':['< 1 month','1-< 2 month','2-< 6 month','6 months and older'], 
                                    'hr_age_factor':[3.833, 3.766, 3.733, 3.533], 
                                    'rr_age_factor':[0.933, 0.9, 0.866, 0.8]}, 
                                    index=[1,2,3,4]) 

# setting for mp
np.seterr(divide='ignore', invalid='ignore')


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


def max_recovery(data):
    r = []

    for i in range(len(data)):
        for j in range(i+1, len(data)):
            r.append(data[j] - data[i])

    return max(r)


@labeling_function()
def lf_immediate_recovery_10s(x):
    """
    If SpO2 level increases/recovers by more than 20 percentage points within 
    10 seconds of alarm start then the alarm is suppressible, otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    spo2_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=10), ['SPO2-%']].to_numpy().flatten()
    return SUPPRESSIBLE if max_recovery(spo2_subrange) > 20 else ABSTAIN


@labeling_function()
def lf_immediate_recovery_15s(x):
    """
    If SpO2 level increases/recovers by more than 30 percentage points within 
    15 seconds of alarm start then the alarm is suppressible, otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    spo2_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=15), ['SPO2-%']].to_numpy().flatten()
    return SUPPRESSIBLE if max_recovery(spo2_subrange) > 30 else ABSTAIN


@labeling_function()
def lf_hr_tech_err_20(x):
    """
    If the difference between the SpO2 HR and ECG HR is larger than 20 percentage points 
    at time of alarm then suppressible, otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    t = v_df.index.get_loc(x.alarm_datetime, method='bfill')  # or use 'nearest' to get closest timestamp in general
    return SUPPRESSIBLE if abs(v_df.at[t,'SPO2-R'] - v_df.at[t,'HR']) > 20 else ABSTAIN


@labeling_function()
def lf_hr_tech_err_30(x):
    """
    If the difference between the SpO2 HR and ECG HR is larger than 30 percentage points 
    at time of alarm then suppressible, otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    t = v_df.index.get_loc(x.alarm_datetime, method='bfill')  # or use 'nearest' to get closest timestamp in general
    return SUPPRESSIBLE if abs(v_df.at[t,'SPO2-R'] - v_df.at[t,'HR']) > 30 else ABSTAIN


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
    age_factor = age_factors_df.hr_age_factor[x.pt_age_group]
    return NOT_SUPPRESSIBLE if np.all( (40*age_factor < hr_subrange) & (hr_subrange <= 50*age_factor) ) else ABSTAIN


@labeling_function()
def lf_hr_below40_over60s(x):
    """
    If HR below 40 * age factor for over 60 seconds since alarm start then the alarm is not suppressible, 
    otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    hr_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=61), ['HR']]
    age_factor = age_factors_df.hr_age_factor[x.pt_age_group]
    return NOT_SUPPRESSIBLE if np.all( (30*age_factor < hr_subrange) & (hr_subrange <= 40*age_factor) ) else ABSTAIN


@labeling_function()
def lf_hr_below30(x):
    """
    If HR below 30 * age factor for any duration since alarm start then the alarm is not suppressible, 
    otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    hr_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=x.duration), ['HR']]
    age_factor = age_factors_df.hr_age_factor[x.pt_age_group]
    return NOT_SUPPRESSIBLE if np.any( hr_subrange <= 30*age_factor ) else ABSTAIN


@labeling_function()
def lf_rr_below50_over120s(x):
    """
    If RR below 50 * age factor for over 120 seconds since alarm start then the alarm is not suppressible, 
    otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    rr_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=121), ['RESP']]
    age_factor = age_factors_df.rr_age_factor[x.pt_age_group]
    return NOT_SUPPRESSIBLE if np.all( (40*age_factor < rr_subrange) & (rr_subrange <= 50*age_factor) ) else ABSTAIN


@labeling_function()
def lf_rr_below40_over60s(x):
    """
    If RR below 40 * age factor for over 60 seconds since alarm start then the alarm is not suppressible, 
    otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    rr_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=61), ['RESP']]
    age_factor = age_factors_df.rr_age_factor[x.pt_age_group]
    return NOT_SUPPRESSIBLE if np.all( (30*age_factor < rr_subrange) & (rr_subrange <= 40*age_factor) ) else ABSTAIN


@labeling_function()
def lf_rr_below30(x):
    """
    If RR below 30 * age factor for any duration since alarm start then the alarm is not suppressible, 
    otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    rr_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=x.duration), ['RESP']]
    age_factor = age_factors_df.rr_age_factor[x.pt_age_group]
    return NOT_SUPPRESSIBLE if np.any( rr_subrange <= 30*age_factor ) else ABSTAIN


def repeat_alarms(x, t):
    """
    If there exists other SpO2 alarms 't' minutes prior to the current alarm's start time and/or
    if there exists other SpO2 alarms 't' minutes after the current alarm's end time then
    the alarm is not suppressible, otherwise abstain

    Args:
        x - alarm instance
        t - timespan to consider (in seconds)
    """
    prior_alarms = alarms_df[ (alarms_df['pt_id'] == x.pt_id) & \
                                (x.alarm_datetime - timedelta(seconds=t) <= alarms_df['alarm_datetime']) & \
                                (alarms_df['alarm_datetime'] < x.alarm_datetime) ]

    subsq_alarms = alarms_df[ (alarms_df['pt_id'] == x.pt_id) & \
                                (x.alarm_datetime + timedelta(seconds=int(x.duration)) <= alarms_df['alarm_datetime']) & \
                                (alarms_df['alarm_datetime'] <= x.alarm_datetime + timedelta(seconds=int(x.duration + t))) ]
    
    count = prior_alarms.shape[0] + subsq_alarms.shape[0]
    
    return NOT_SUPPRESSIBLE if count > 0 else ABSTAIN


@labeling_function()
def lf_repeat_alarms_15s(x):
    """
    If there exists other alarms within 15 seconds of the current alarm then the alarm is
    not suppressible, otherwise abstain
    """
    return repeat_alarms(x, 15) 


@labeling_function()
def lf_repeat_alarms_30s(x):
    """
    If there exists other alarms within 30 seconds of the current alarm then the alarm is
    not suppressible, otherwise abstain
    """
    return repeat_alarms(x, 30) 


@labeling_function()
def lf_repeat_alarms_60s(x):
    """
    If there exists other alarms within 60 seconds of the current alarm then the alarm is
    not suppressible, otherwise abstain
    """
    return repeat_alarms(x, 60) 


@labeling_function()
def lf_outlier_spo2_120(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=60)):(x.alarm_datetime + timedelta(seconds=60)), 'SPO2-%_mp120']
    return SUPPRESSIBLE if np.any(sub_interval > 8.4) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_110(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=55)):(x.alarm_datetime + timedelta(seconds=55)), 'SPO2-%_mp110']
    return SUPPRESSIBLE if np.any(sub_interval > 7.8) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_100(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=50)):(x.alarm_datetime + timedelta(seconds=50)), 'SPO2-%_mp100']
    return SUPPRESSIBLE if np.any(sub_interval > 7.2) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_90(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=45)):(x.alarm_datetime + timedelta(seconds=45)), 'SPO2-%_mp90']
    return SUPPRESSIBLE if np.any(sub_interval > 6.6) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_80(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=40)):(x.alarm_datetime + timedelta(seconds=40)), 'SPO2-%_mp80']
    return SUPPRESSIBLE if np.any(sub_interval > 6.0) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_70(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=35)):(x.alarm_datetime + timedelta(seconds=35)), 'SPO2-%_mp70']
    return SUPPRESSIBLE if np.any(sub_interval > 5.3) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_60(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=30)):(x.alarm_datetime + timedelta(seconds=30)), 'SPO2-%_mp60']
    return SUPPRESSIBLE if np.any(sub_interval > 4.6) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_50(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=25)):(x.alarm_datetime + timedelta(seconds=25)), 'SPO2-%_mp50']
    return SUPPRESSIBLE if np.any(sub_interval > 3.8) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_40(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=20)):(x.alarm_datetime + timedelta(seconds=20)), 'SPO2-%_mp40']
    return SUPPRESSIBLE if np.any(sub_interval > 2.9) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_30(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=15)):(x.alarm_datetime + timedelta(seconds=15)), 'SPO2-%_mp30']
    return SUPPRESSIBLE if np.any(sub_interval > 2.1) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_20(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=10)):(x.alarm_datetime + timedelta(seconds=10)), 'SPO2-%_mp20']
    return SUPPRESSIBLE if np.any(sub_interval > 1.0) else ABSTAIN


@labeling_function()
def lf_outlier_hr_120(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=60)):(x.alarm_datetime + timedelta(seconds=60)), 'HR_mp120']
    return SUPPRESSIBLE if np.any(sub_interval > 9.0) else ABSTAIN


@labeling_function()
def lf_outlier_hr_110(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=55)):(x.alarm_datetime + timedelta(seconds=55)), 'HR_mp110']
    return SUPPRESSIBLE if np.any(sub_interval > 8.5) else ABSTAIN


@labeling_function()
def lf_outlier_hr_100(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=50)):(x.alarm_datetime + timedelta(seconds=50)), 'HR_mp100']
    return SUPPRESSIBLE if np.any(sub_interval > 7.8) else ABSTAIN


@labeling_function()
def lf_outlier_hr_90(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=45)):(x.alarm_datetime + timedelta(seconds=45)), 'HR_mp90']
    return SUPPRESSIBLE if np.any(sub_interval > 7.3) else ABSTAIN


@labeling_function()
def lf_outlier_hr_80(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=40)):(x.alarm_datetime + timedelta(seconds=40)), 'HR_mp80']
    return SUPPRESSIBLE if np.any(sub_interval > 6.7) else ABSTAIN


@labeling_function()
def lf_outlier_hr_70(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=35)):(x.alarm_datetime + timedelta(seconds=35)), 'HR_mp70']
    return SUPPRESSIBLE if np.any(sub_interval > 6.0) else ABSTAIN


@labeling_function()
def lf_outlier_hr_60(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=30)):(x.alarm_datetime + timedelta(seconds=30)), 'HR_mp60']
    return SUPPRESSIBLE if np.any(sub_interval > 5.4) else ABSTAIN


@labeling_function()
def lf_outlier_hr_50(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=25)):(x.alarm_datetime + timedelta(seconds=25)), 'HR_mp50']
    return SUPPRESSIBLE if np.any(sub_interval > 4.7) else ABSTAIN


@labeling_function()
def lf_outlier_hr_40(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=20)):(x.alarm_datetime + timedelta(seconds=20)), 'HR_mp40']
    return SUPPRESSIBLE if np.any(sub_interval > 3.9) else ABSTAIN


@labeling_function()
def lf_outlier_hr_30(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=15)):(x.alarm_datetime + timedelta(seconds=15)), 'HR_mp30']
    return SUPPRESSIBLE if np.any(sub_interval > 3.1) else ABSTAIN


@labeling_function()
def lf_outlier_hr_20(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=10)):(x.alarm_datetime + timedelta(seconds=10)), 'HR_mp20']
    return SUPPRESSIBLE if np.any(sub_interval > 2.1) else ABSTAIN


@labeling_function()
def lf_outlier_rr_120(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=60)):(x.alarm_datetime + timedelta(seconds=60)), 'RESP_mp120']
    return SUPPRESSIBLE if np.any(sub_interval > 8.7) else ABSTAIN


@labeling_function()
def lf_outlier_rr_110(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=55)):(x.alarm_datetime + timedelta(seconds=55)), 'RESP_mp110']
    return SUPPRESSIBLE if np.any(sub_interval > 8.1) else ABSTAIN


@labeling_function()
def lf_outlier_rr_100(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=50)):(x.alarm_datetime + timedelta(seconds=50)), 'RESP_mp100']
    return SUPPRESSIBLE if np.any(sub_interval > 7.6) else ABSTAIN


@labeling_function()
def lf_outlier_rr_90(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=45)):(x.alarm_datetime + timedelta(seconds=45)), 'RESP_mp90']
    return SUPPRESSIBLE if np.any(sub_interval > 7.1) else ABSTAIN


@labeling_function()
def lf_outlier_rr_80(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=40)):(x.alarm_datetime + timedelta(seconds=40)), 'RESP_mp80']
    return SUPPRESSIBLE if np.any(sub_interval > 6.5) else ABSTAIN


@labeling_function()
def lf_outlier_rr_70(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=35)):(x.alarm_datetime + timedelta(seconds=35)), 'RESP_mp70']
    return SUPPRESSIBLE if np.any(sub_interval > 6.0) else ABSTAIN


@labeling_function()
def lf_outlier_rr_60(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=30)):(x.alarm_datetime + timedelta(seconds=30)), 'RESP_mp60']
    return SUPPRESSIBLE if np.any(sub_interval > 5.4) else ABSTAIN


@labeling_function()
def lf_outlier_rr_50(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=25)):(x.alarm_datetime + timedelta(seconds=25)), 'RESP_mp50']
    return SUPPRESSIBLE if np.any(sub_interval > 4.7) else ABSTAIN


@labeling_function()
def lf_outlier_rr_40(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=20)):(x.alarm_datetime + timedelta(seconds=20)), 'RESP_mp40']
    return SUPPRESSIBLE if np.any(sub_interval > 3.9) else ABSTAIN


@labeling_function()
def lf_outlier_rr_30(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=15)):(x.alarm_datetime + timedelta(seconds=15)), 'RESP_mp30']
    return SUPPRESSIBLE if np.any(sub_interval > 3.0) else ABSTAIN


@labeling_function()
def lf_outlier_rr_20(x):
    v_df = vitals_dfs[x.pt_id]
    sub_interval = v_df.loc[(x.alarm_datetime - timedelta(seconds=10)):(x.alarm_datetime + timedelta(seconds=10)), 'RESP_mp20']
    return SUPPRESSIBLE if np.any(sub_interval > 2.0) else ABSTAIN


# compute moving std
def compute_std(ts, window):
    ts_std = []
    for i in range(len(ts) - window):
        ts_std.append(np.std(ts[i: i+window-1]))
    ts_std = np.array(ts_std)
    ts_std_head = np.zeros(window // 2 - 1)
    ts_std_tail = np.zeros(len(ts) - len(ts_std) - window // 2 + 1)
    ts_std = np.concatenate([ts_std_head, ts_std, ts_std_tail])
    return ts_std


# compute matrix profile at given window and threshold, return mp and percentage of outliers
def compute_mp(ts, window, threshold):
    # sfp - commenting this out as it is unneccesary
    # remove trailing nans of ts
    #i = len(ts) - 1
    #while np.isnan(ts[i]) and i >= 0:
    #    i -= 1
    #ts = ts[0:i+1]
    # compute mp by stamp
    mp = np.array(matrixProfile.stomp(ts, m=window))[0]
    # calibrate ts and mp, so mp value is assigned to the middle of that window
    mp_head = np.zeros(window//2 - 1)
    mp_tail = np.zeros(len(ts) - len(mp) - window//2 + 1)
    mp = np.concatenate([mp_head, mp, mp_tail])
    # remove error results due to zero std (make them 0 so they don't contribute to outliers)
    ts_std = compute_std(ts, window=window)
    count_zero_std = 0
    for i in range(len(ts_std)):
        if ts_std[i] == 0:
            mp[i] = 0
            count_zero_std += 1
    # sfp - commenting this out because not relevant to lfs
    # compute percentage of outliers, where head, tail and zero std points do not participate
    #outlier = mp[np.where(mp > threshold)]
    #outlier_percentage = len(outlier) / (len(mp) - len(mp_head) - len(mp_tail) - count_zero_std)
    return mp   #, outlier_percentage


def compute_all_mps(verbose=True):
    # parameters for matrix profile calculations
    s = [(120,8.4), (110,7.8), (100,7.2), (90,6.6), (80,6.0), (70,5.3), (60,4.6), (50,3.8), (40,2.9), (30,2.1), (20,1.0)]
    h = [(120,9.0), (110,8.5), (100,7.8), (90,7.3), (80,6.7), (70,6.0), (60,5.4), (50,4.7), (40,3.9), (30,3.1), (20,2.1)]
    r = [(120,8.7), (110,8.1), (100,7.6), (90,7.1), (80,6.5), (70,6.0), (60,5.4), (50,4.7), (40,3.9), (30,3.0), (20,2.0)]
    
    for pt_id, v_df in vitals_dfs.items():
        if verbose:
            print('Computing mps for pt ' + str(pt_id))
        
        for x in s:
            if 'SPO2-%' in v_df.columns:
                v_df['SPO2-%_mp' + str(x[0])] = compute_mp(v_df['SPO2-%'].to_numpy(), window=x[0], threshold=x[1])
        for x in h:
            if 'HR' in v_df.columns:
                v_df['HR_mp' + str(x[0])] = compute_mp(v_df['HR'].to_numpy(), window=x[0], threshold=x[1])
        for x in r:
            if 'RESP' in v_df.columns:
                v_df['RESP_mp' + str(x[0])] = compute_mp(v_df['RESP'].to_numpy(), window=x[0], threshold=x[1])


def pred_threshold(p, thres):
    """
    Predict alarm is SUPPRESSIBLE iff probability of suppressible is larger 
    than given threshold and is larger than probability of not-suppressible

    Args:
        p - array [P(NOT_SUPPRESSIBLE), P(SUPPRESSIBLE)]
        thres - threshold for SUPPRESSIBLE prediction

    Return:
        int - 1 if suppressible, 0 if not-suppressible, -1 if abstain
    """
    if p[SUPPRESSIBLE] > p[NOT_SUPPRESSIBLE] and p[SUPPRESSIBLE] >= thres:
        return SUPPRESSIBLE
    elif p[NOT_SUPPRESSIBLE] > p[SUPPRESSIBLE]:
        return NOT_SUPPRESSIBLE
    else:
        return ABSTAIN


def main():
    # computing matrix profiles
    compute_all_mps()

    ## Apply LFs to all alarms
    lfs = [
            lf_short_alarm_15s, lf_short_alarm_10s, lf_short_alarm_5s,
            lf_long_alarm_60s, lf_long_alarm_65s, lf_long_alarm_70s,
            lf_hr_tech_err_20, lf_hr_tech_err_30,
            lf_spo2_below85_over120s, lf_spo2_below80_over100s, lf_spo2_below70_over90s, lf_spo2_below60_over60s, lf_spo2_below50_over30s,
            lf_hr_below50_over120s, lf_hr_below40_over60s, lf_hr_below30,
            lf_rr_below50_over120s, lf_rr_below40_over60s, lf_rr_below30,
            lf_repeat_alarms_15s, lf_repeat_alarms_30s, lf_repeat_alarms_60s,
            lf_outlier_spo2_120, lf_outlier_spo2_110, lf_outlier_spo2_100, lf_outlier_spo2_90, lf_outlier_spo2_80, lf_outlier_spo2_70, lf_outlier_spo2_60, lf_outlier_spo2_50, lf_outlier_spo2_40, lf_outlier_spo2_30, lf_outlier_spo2_20,
            lf_outlier_hr_120, lf_outlier_hr_110, lf_outlier_hr_100, lf_outlier_hr_90, lf_outlier_hr_80, lf_outlier_hr_70, lf_outlier_hr_60, lf_outlier_hr_50, lf_outlier_hr_40, lf_outlier_hr_30, lf_outlier_hr_20,
            lf_outlier_rr_120, lf_outlier_rr_110, lf_outlier_rr_100, lf_outlier_rr_90, lf_outlier_rr_80, lf_outlier_rr_70, lf_outlier_rr_60, lf_outlier_rr_50, lf_outlier_rr_40, lf_outlier_rr_30, lf_outlier_rr_20
        ]

    print('Applying LFs...\n')
    applier = PandasParallelLFApplier(lfs)
    L_train = applier.apply(alarms_df, n_parallel=50, scheduler='threads', fault_tolerant=True)
    print(LFAnalysis(L_train, lfs).lf_summary(Y=alarms_df.true_label.to_numpy()), '\n')


    ## Train a label model
    label_model = LabelModel(cardinality=2)
    label_model.fit(L_train)
    #label_model.save('label_model_' + datetime.now() + '.pkl')


    ## Evaluate label model accuracy
    # original prediction method
    pred = label_model.predict(L_train)
    print('Confusion Matrix (original): ')
    print(confusion_matrix(alarms_df.true_label, pred))

    # predict with threshold
    pred_probs = label_model.predict_proba(L_train)

    for thres in [0.7,0.8,0.9,0.95,0.99]:
        pred = np.apply_along_axis(pred_threshold, 1, pred_probs, thres)
        print('Confusion Matrix (threshold=' + str(thres) + '):')
        print( confusion_matrix(alarms_df.true_label, pred) )


def test_lf():
    x = alarms_df.iloc[194]
    print(x)

    compute_all_mps()
    v_df = vitals_dfs[505]
    print(v_df)


if __name__ == '__main__':
    #main()
    test_lf()
