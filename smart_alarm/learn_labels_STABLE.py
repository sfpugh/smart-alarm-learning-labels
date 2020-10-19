#!/usr/bin/env python
# coding: utf-8

# # Smart Alarm 2.0: Learning SpO2 Alarm Labels
# 
# Smart Alarm 2.0 is concerned with reducing the number of false SpO2 low alarms produced by pulse oximetry machines. We intend to use some machine learning techniques in order to make a "smart" alarm that will automatically suppress false alarms. A big challenge with ML is that large labeled datasets are unavailable and expensive to produce. Hence, here we are experimenting with the Snorkel data programming paradigm in aims of developing a probabilistic label generating model for SpO2 alarms.
# 
# To do eventually:
# - [ ] Experiment with prec_init (cant do this until bug patch, see https://github.com/snorkel-team/snorkel/issues/1484)
# - [ ] Indicate interval/sampling-rate per patient (e.g., patient 501 has 60sec interval, whereas others have 5sec)
# - [ ] Edit excel/csv file for pt 580
# 

# In[ ]:


EXAMPLE = "smart_alarm"


# In[ ]:

import numpy as np
import pandas as pd
from snorkel.labeling import labeling_function, LFAnalysis
from snorkel.labeling.apply.dask import PandasParallelLFApplier
from datetime import datetime, timedelta


# In[ ]:


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
np.set_printoptions(threshold=np.inf)


# ## Initialization and Data Loading
# - **alarms_df**: data for 3,265 SpO2 low alarms from CHOP data
# - **vitals_df**: data for vital signs of 100 patients related to the alarms data

# In[ ]:


# For clarity, defining constants for to represent class labels

ABSTAIN = -1
NOT_SUPPRESSIBLE = 0
SUPPRESSIBLE = 1


# In[ ]:


# Setting global constant to represent sampling rate of 1 data point per 5 seconds in CHOP data

INTERVAL = 5


# In[ ]:


# Seed for randomized functions

SEED = 42


# The following data is from the CHOP dataset that Chris provided.

# In[ ]:


from os import path
from read_chop_data import read_alarms, read_vitals
from compute_mp import apply_compute_mp

#prefix = "data/"
prefix = "/data2/sfpugh/smart-alarm-2.0/chop-alarm-data/"    # if on Ash02, the files already exist there

# Read in alarms data
if path.exists(prefix + "chop_spo2alarms_df.pkl"):
    alarms_df = pd.read_pickle(prefix + "chop_spo2alarms_df.pkl")
else:
    alarms_df = read_alarms(save=True)

# Read in vitals data
if path.exists(prefix + "chop_vitals_with_mp_df.pkl"):
    vitals_df = pd.read_pickle(prefix + "chop_vitals_with_mp_df.pkl")    # CHOP vitals with precomputed matrix profiles
elif path.exists(prefix + "chop_vitals_df.pkl"):
    temp_vitals_df = pd.read_pickle(prefix + "chop_vitals_df.pkl")
    vitals_df = apply_compute_mp(vitals_wo_mp_df, ["SPO2-%","HR","RESP"], INTERVAL, save=True, filename=prefix + "chop_vitals_with_mp_df.pkl")
else:
    temp_vitals_df = read_vitals(verbose=True, save=True)
    vitals_df = apply_compute_mp(temp_vitals_df, ["SPO2-%","HR","RESP"], INTERVAL, save=True, filename="chop_vitals_with_mp_df.pkl")


# In[ ]:


# Set age factors for labeling functions

age_factors_df = pd.DataFrame({"pt_age_group":[1,2,3,4], 
                                "pt_age_group_L":["< 1 month","1-< 2 month","2-< 6 month","6 months and older"], 
                                "hr_age_factor":[3.833, 3.766, 3.733, 3.533], 
                                "rr_age_factor":[0.933, 0.9, 0.866, 0.8]}, 
                                index=[1,2,3,4])


# ## Define Labeling Functions
# The following labeling functions are translations of rules received from clinicians on what describes NOT_SUPPRESSIBLE and SUPPRESSIBLE SpO2 alarms. See https://docs.google.com/spreadsheets/d/1_1QBVaiWl4SkBy9vEFBk5Uv0HfPWUMX8Sg_IPZoJbfA/edit?usp=sharing.

# In[ ]:


def get_vitals(pt_id, v_sign, t_start=None, t_end=None):
    """
    Get timeseries of a specific vital sign for a given patient
    
    Args:
        pt_id - integer id of patient
        v_sign - string vital sign name
        t_start - start timestamp 
        t_end - end timestamp
        
    Return:
        timeseries array of vital sign for patient
    """
    return vitals_df.loc[(pt_id, ), v_sign][t_start:t_end]


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
def lf_spo2_below80_over120s(x):
    """
    If SpO2 level is less than 80 for longer than 120 seconds since alarm start
    then the alarm is not suppressible, otherwise abstain
    """
    spo2 = get_vitals(x.pt_id, "SPO2-%", t_start=x.alarm_datetime, t_end=(x.alarm_datetime + timedelta(seconds=121)))
    return NOT_SUPPRESSIBLE if np.all( spo2 < 80 ) else ABSTAIN

@labeling_function()
def lf_spo2_below85_over120s(x):
    """
    If SpO2 level stays within range (80,85] for longer than 120 seconds since alarm start 
    then the alarm is not suppressible, otherwise abstain
    """
    spo2 = get_vitals(x.pt_id, "SPO2-%", t_start=x.alarm_datetime, t_end=(x.alarm_datetime + timedelta(seconds=121)))
    return NOT_SUPPRESSIBLE if np.all( (80 < spo2) & (spo2 <= 85) ) else ABSTAIN


@labeling_function()
def lf_spo2_below80_over100s(x):
    """
    If SpO2 level stays within range (70,80] for longer than 100 seconds since alarm start 
    then the alarm is not suppressible, otherwise abstain
    """
    spo2 = get_vitals(x.pt_id, "SPO2-%", t_start=x.alarm_datetime, t_end=(x.alarm_datetime + timedelta(seconds=101)))
    return NOT_SUPPRESSIBLE if np.all( (70 < spo2) & (spo2 <= 80) ) else ABSTAIN


@labeling_function()
def lf_spo2_below70_over90s(x):
    """
    If SpO2 level stays within range (60,70] for longer than 90 seconds since alarm start 
    then the alarm is not suppressible, otherwise abstain
    """
    spo2 = get_vitals(x.pt_id, "SPO2-%", t_start=x.alarm_datetime, t_end=(x.alarm_datetime + timedelta(seconds=91)))
    return NOT_SUPPRESSIBLE if np.all( (60 < spo2) & (spo2 <= 70) ) else ABSTAIN


@labeling_function()
def lf_spo2_below60_over60s(x):
    """
    If SpO2 level stays within range (50,60] for longer than 60 seconds since alarm start 
    then the alarm is not suppressible, otherwise abstain
    """
    spo2 = get_vitals(x.pt_id, "SPO2-%", t_start=x.alarm_datetime, t_end=(x.alarm_datetime + timedelta(seconds=61)))
    return NOT_SUPPRESSIBLE if np.all( (50 < spo2) & (spo2 <= 60) ) else ABSTAIN


@labeling_function()
def lf_spo2_below50_over30s(x):
    """
    If SpO2 level stays within range (0,50] for longer than 30 seconds since alarm start 
    then the alarm is not suppressible, otherwise abstain
    """
    spo2 = get_vitals(x.pt_id, "SPO2-%", t_start=x.alarm_datetime, t_end=(x.alarm_datetime + timedelta(seconds=31)))
    return NOT_SUPPRESSIBLE if np.all( spo2 <= 50 ) else ABSTAIN
    

@labeling_function()
def lf_hr_above220_over10s(x):
    """
    If HR is above 220 regardless of age for over 10 seconds since alarm start then the alarm is
    not suppressible, otherwise abstain
    """
    hr = get_vitals(x.pt_id, "HR", t_start=x.alarm_datetime, t_end=(x.alarm_datetime + timedelta(seconds=11)))
    return NOT_SUPPRESSIBLE if np.all( hr > 220 ) else ABSTAIN


@labeling_function()
def lf_hr_below50_over10s(x):
    """
    If HR is below 50 regardless of age for over 10 seconds since alarm start then the alarm is
    not suppressible, otherwise abstain
    """
    hr = get_vitals(x.pt_id, "HR", t_start=x.alarm_datetime, t_end=(x.alarm_datetime + timedelta(seconds=11)))
    return NOT_SUPPRESSIBLE if np.all( hr < 50 ) else ABSTAIN


@labeling_function()
def lf_hr_below50_over120s(x):
    """
    If HR below 50 * age factor for over 120 seconds since alarm start then the alarm is not suppressible, 
    otherwise abstain
    """
    hr = get_vitals(x.pt_id, "HR", t_start=x.alarm_datetime, t_end=(x.alarm_datetime + timedelta(seconds=121)))
    age_factor = age_factors_df.loc[x.pt_age_group, "hr_age_factor"]
    return NOT_SUPPRESSIBLE if np.all( (40*age_factor < hr) & (hr <= 50*age_factor) ) else ABSTAIN
    
    
@labeling_function()
def lf_hr_below40_over60s(x):
    """
    If HR below 40 * age factor for over 60 seconds since alarm start then the alarm is not suppressible, 
    otherwise abstain
    """
    hr = get_vitals(x.pt_id, "HR", t_start=x.alarm_datetime, t_end=(x.alarm_datetime + timedelta(seconds=61)))
    age_factor = age_factors_df.loc[x.pt_age_group, "hr_age_factor"]
    return NOT_SUPPRESSIBLE if np.all( (30*age_factor < hr) & (hr <= 40*age_factor) ) else ABSTAIN


@labeling_function()
def lf_hr_below30(x):
    """
    If HR below 30 * age factor for any duration since alarm start then the alarm is not suppressible, 
    otherwise abstain
    """
    hr = get_vitals(x.pt_id, "HR", t_start=x.alarm_datetime, t_end=(x.alarm_datetime + timedelta(seconds=int(x.duration))))
    age_factor = age_factors_df.loc[x.pt_age_group, "hr_age_factor"]
    return NOT_SUPPRESSIBLE if np.all( hr <= 30*age_factor ) else ABSTAIN


@labeling_function()
def lf_rr_below10_over60s(x):
    """
    If RR below 10 regardless of patient age for longer than 60 seconds then the alarm is not suppressible,
    otherwise abstain
    """
    rr = get_vitals(x.pt_id, "RESP", t_start=x.alarm_datetime, t_end=(x.alarm_datetime + timedelta(seconds=61))) 
    return NOT_SUPPRESSIBLE if np.all( rr < 10 ) else ABSTAINs


@labeling_function()
def lf_rr_below50_over120s(x):
    """
    If RR below 50 * age factor for over 120 seconds since alarm start then the alarm is not suppressible, 
    otherwise abstain
    """
    rr = get_vitals(x.pt_id, "RESP", t_start=x.alarm_datetime, t_end=(x.alarm_datetime + timedelta(seconds=121)))
    age_factor = age_factors_df.loc[x.pt_age_group, "rr_age_factor"]
    return NOT_SUPPRESSIBLE if np.all( (40*age_factor < rr) & (rr <= 50*age_factor) ) else ABSTAIN


@labeling_function()
def lf_rr_below40_over60s(x):
    """
    If RR below 40 * age factor for over 60 seconds since alarm start then the alarm is not suppressible, 
    otherwise abstain
    """
    rr = get_vitals(x.pt_id, "RESP", t_start=x.alarm_datetime, t_end=(x.alarm_datetime + timedelta(seconds=61)))
    age_factor = age_factors_df.loc[x.pt_age_group, "rr_age_factor"]
    return NOT_SUPPRESSIBLE if np.all( (30*age_factor < rr) & (rr <= 40*age_factor) ) else ABSTAIN


@labeling_function()
def lf_rr_below30(x):
    """
    If RR below 30 * age factor for any duration since alarm start then the alarm is not suppressible, 
    otherwise abstain
    """
    rr = get_vitals(x.pt_id, "RESP", t_start=x.alarm_datetime, t_end=(x.alarm_datetime + timedelta(seconds=int(x.duration))))
    age_factor = age_factors_df.loc[x.pt_age_group, "rr_age_factor"]
    return NOT_SUPPRESSIBLE if np.all( rr <= 30*age_factor ) else ABSTAIN
    
    
def count_alarms_in_timespan(x, t):
    """
    Count the number of other SpO2 alarms t seconds before the current alarm's start time
    and t seconds after the current alarm's end time
    

    Args:
        x - alarm instance
        t - timespan to consider (in seconds)

    Return:
        int - number of alarms occurred in timespan
    """
    prior_alarms = alarms_df[ (alarms_df["pt_id"] == x.pt_id) &                                 (x.alarm_datetime - timedelta(seconds=t) <= alarms_df["alarm_datetime"]) &                                 (alarms_df["alarm_datetime"] < x.alarm_datetime) ]

    subsq_alarms = alarms_df[ (alarms_df["pt_id"] == x.pt_id) &                                 (x.alarm_datetime + timedelta(seconds=int(x.duration)) <= alarms_df["alarm_datetime"]) &                                 (alarms_df["alarm_datetime"] <= x.alarm_datetime + timedelta(seconds=int(x.duration + t))) ]
    
    return prior_alarms.shape[0] + subsq_alarms.shape[0]
    

@labeling_function()
def lf_10alarms_5mins(x):
    """
    If there are more than 10 alarms within 5 minutes of the current alarm then the alarm is
    not suppressible, otherwise abstain
    """
    return NOT_SUPPRESSIBLE if count_alarms_in_timespan(x, 300) > 10 else ABSTAIN


@labeling_function()
def lf_repeat_alarms_15s(x):
    """
    If there exists other alarms within 15 seconds of the current alarm then the alarm is
    not suppressible, otherwise abstain
    """
    return NOT_SUPPRESSIBLE if count_alarms_in_timespan(x, 15) > 0 else ABSTAIN


@labeling_function()
def lf_repeat_alarms_30s(x):
    """
    If there exists other alarms within 30 seconds of the current alarm then the alarm is
    not suppressible, otherwise abstain
    """
    return NOT_SUPPRESSIBLE if count_alarms_in_timespan(x, 30) > 0 else ABSTAIN


@labeling_function()
def lf_repeat_alarms_60s(x):
    """
    If there exists other alarms within 60 seconds of the current alarm then the alarm is
    not suppressible, otherwise abstain
    """
    return NOT_SUPPRESSIBLE if count_alarms_in_timespan(x, 60) > 0 else ABSTAIN


@labeling_function()
def lf_gradual_recovery(x):
    """
    If SpO2 increases at 5-10% per minute after the alarm then the alarm is not suppressible,
    otherwise abstain
    """
    spo2 = get_vitals(x.pt_id, "SPO2-%")
    return ABSTAIN


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
    """
    Determine the maximum recovery between two consecutive data points in given data
    
    Args:
        data - array of numeric measurements
        
    Return:
        float - maximum recovery
    """
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
    spo2 = get_vitals(x.pt_id, "SPO2-%", t_start=x.alarm_datetime, t_end=(x.alarm_datetime + timedelta(seconds=10)))
    if len(spo2) < 2:
        return ABSTAIN
    return SUPPRESSIBLE if max_recovery(spo2) > 20 else ABSTAIN


@labeling_function()
def lf_immediate_recovery_15s(x):
    """
    If SpO2 level increases/recovers by more than 30 percentage points within 
    15 seconds of alarm start then the alarm is suppressible, otherwise abstain
    """
    spo2 = get_vitals(x.pt_id, "SPO2-%", t_start=x.alarm_datetime, t_end=(x.alarm_datetime + timedelta(seconds=15)))
    if len(spo2) < 2:
        return ABSTAIN
    return SUPPRESSIBLE if max_recovery(spo2) > 30 else ABSTAIN


@labeling_function()
def lf_hr_tech_err_20(x):
    """
    If the difference between the SpO2 HR and ECG HR is larger than 20 percentage points 
    at time of alarm then suppressible, otherwise abstain
    """
    spo2_hr = get_vitals(x.pt_id, "SPO2-R")
    ecg_hr = get_vitals(x.pt_id, "HR")
    return SUPPRESSIBLE if abs(spo2_hr[x.alarm_datetime] - ecg_hr[x.alarm_datetime]) > 20 else ABSTAIN


@labeling_function()
def lf_hr_tech_err_30(x):
    """
    If the difference between the SpO2 HR and ECG HR is larger than 30 percentage points 
    at time of alarm then suppressible, otherwise abstain
    """
    spo2_hr = get_vitals(x.pt_id, "SPO2-R")
    ecg_hr = get_vitals(x.pt_id, "HR")
    return SUPPRESSIBLE if abs(spo2_hr[x.alarm_datetime] - ecg_hr[x.alarm_datetime]) > 30 else ABSTAIN


@labeling_function()
def lf_outlier_spo2_120(x):
    """
    If there exists an outlier (spike in matrix profile larger than 8.4) in the 
    120 second window surrounding the time of alarm then suppressible, otherwise abstain
    """
    spo2 = get_vitals(x.pt_id, 
                      "SPO2-% MP120",
                      t_start=(x.alarm_datetime - timedelta(seconds=60)),
                      t_end=(x.alarm_datetime + timedelta(seconds=60)) )
    return SUPPRESSIBLE if np.any( spo2 > 8.4 ) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_110(x):
    """
    If there exists an outlier (spike in matrix profile larger than 7.8) in the 
    110 second window surrounding the time of alarm then suppressible, otherwise abstain
    """
    spo2 = get_vitals(x.pt_id, 
                      "SPO2-% MP110",
                      t_start=(x.alarm_datetime - timedelta(seconds=55)),
                      t_end=(x.alarm_datetime + timedelta(seconds=55)) )
    return SUPPRESSIBLE if np.any( spo2 > 7.8 ) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_100(x):
    """
    If there exists an outlier (spike in matrix profile larger than 7.2) in the 
    100 second window surrounding the time of alarm then suppressible, otherwise abstain
    """
    spo2 = get_vitals(x.pt_id, 
                      "SPO2-% MP100",
                      t_start=(x.alarm_datetime - timedelta(seconds=50)),
                      t_end=(x.alarm_datetime + timedelta(seconds=50)) )
    return SUPPRESSIBLE if np.any( spo2 > 7.2 ) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_90(x):
    """
    If there exists an outlier (spike in matrix profile larger than 6.6) in the 
    90 second window surrounding the time of alarm then suppressible, otherwise abstain
    """
    spo2 = get_vitals(x.pt_id, 
                      "SPO2-% MP90",
                      t_start=(x.alarm_datetime - timedelta(seconds=45)),
                      t_end=(x.alarm_datetime + timedelta(seconds=45)) )
    return SUPPRESSIBLE if np.any( spo2 > 6.6 ) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_80(x):
    """
    If there exists an outlier (spike in matrix profile larger than 6.0) in the 
    80 second window surrounding the time of alarm then suppressible, otherwise abstain
    """
    spo2 = get_vitals(x.pt_id, 
                      "SPO2-% MP80",
                      t_start=(x.alarm_datetime - timedelta(seconds=40)),
                      t_end=(x.alarm_datetime + timedelta(seconds=40)) )
    return SUPPRESSIBLE if np.any( spo2 > 6.0 ) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_70(x):
    """
    If there exists an outlier (spike in matrix profile larger than 5.3) in the 
    70 second window surrounding the time of alarm then suppressible, otherwise abstain
    """
    spo2 = get_vitals(x.pt_id, 
                      "SPO2-% MP70",
                      t_start=(x.alarm_datetime - timedelta(seconds=35)),
                      t_end=(x.alarm_datetime + timedelta(seconds=35)) )
    return SUPPRESSIBLE if np.any( spo2 > 5.3 ) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_60(x):
    """
    If there exists an outlier (spike in matrix profile larger than 4.6) in the 
    60 second window surrounding the time of alarm then suppressible, otherwise abstain
    """
    spo2 = get_vitals(x.pt_id, 
                      "SPO2-% MP60",
                      t_start=(x.alarm_datetime - timedelta(seconds=30)),
                      t_end=(x.alarm_datetime + timedelta(seconds=30)) )
    return SUPPRESSIBLE if np.any( spo2 > 4.6 ) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_50(x):
    """
    If there exists an outlier (spike in matrix profile larger than 3.8) in the 
    50 second window surrounding the time of alarm then suppressible, otherwise abstain
    """
    spo2 = get_vitals(x.pt_id, 
                      "SPO2-% MP50",
                      t_start=(x.alarm_datetime - timedelta(seconds=25)),
                      t_end=(x.alarm_datetime + timedelta(seconds=25)) )
    return SUPPRESSIBLE if np.any( spo2 > 3.8 ) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_40(x):
    """
    If there exists an outlier (spike in matrix profile larger than 2.9) in the 
    40 second window surrounding the time of alarm then suppressible, otherwise abstain
    """
    spo2 = get_vitals(x.pt_id, 
                      "SPO2-% MP40",
                      t_start=(x.alarm_datetime - timedelta(seconds=20)),
                      t_end=(x.alarm_datetime + timedelta(seconds=20)) )
    return SUPPRESSIBLE if np.any( spo2 > 2.9 ) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_30(x):
    """
    If there exists an outlier (spike in matrix profile larger than 2.1) in the 
    30 second window surrounding the time of alarm then suppressible, otherwise abstain
    """
    spo2 = get_vitals(x.pt_id, 
                      "SPO2-% MP30",
                      t_start=(x.alarm_datetime - timedelta(seconds=15)),
                      t_end=(x.alarm_datetime + timedelta(seconds=15)) )
    return SUPPRESSIBLE if np.any( spo2 > 2.1 ) else ABSTAIN


@labeling_function()
def lf_outlier_spo2_20(x):
    """
    If there exists an outlier (spike in matrix profile larger than 1.0) in the 
    20 second window surrounding the time of alarm then suppressible, otherwise abstain
    """
    spo2 = get_vitals(x.pt_id, 
                      "SPO2-% MP20",
                      t_start=(x.alarm_datetime - timedelta(seconds=10)),
                      t_end=(x.alarm_datetime + timedelta(seconds=10)) )
    return SUPPRESSIBLE if np.any( spo2 > 1.0 ) else ABSTAIN


@labeling_function()
def lf_outlier_hr_120(x):
    """
    If there exists at least one data point where the 120-second window heart rate matrix profile
    exceeds 9.0 within a 120-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    hr_mp = get_vitals(x.pt_id, 
                       "HR MP120",
                       t_start=(x.alarm_datetime - timedelta(seconds=60)),
                       t_end=(x.alarm_datetime + timedelta(seconds=60)) )
    return SUPPRESSIBLE if np.any( hr_mp > 9.0 ) else ABSTAIN


@labeling_function()
def lf_outlier_hr_110(x):
    """
    If there exists at least one data point where the 110-second window heart rate matrix profile
    exceeds 8.5 within a 110-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    hr_mp = get_vitals(x.pt_id, 
                       "HR MP110",
                       t_start=(x.alarm_datetime - timedelta(seconds=55)),
                       t_end=(x.alarm_datetime + timedelta(seconds=55)) )
    return SUPPRESSIBLE if np.any( hr_mp > 8.5 ) else ABSTAIN


@labeling_function()
def lf_outlier_hr_100(x):
    """
    If there exists at least one data point where the 100-second window heart rate matrix profile
    exceeds 7.8 within a 100-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    hr_mp = get_vitals(x.pt_id, 
                       "HR MP100",
                       t_start=(x.alarm_datetime - timedelta(seconds=50)),
                       t_end=(x.alarm_datetime + timedelta(seconds=50)) )
    return SUPPRESSIBLE if np.any( hr_mp > 7.8 ) else ABSTAIN


@labeling_function()
def lf_outlier_hr_90(x):
    """
    If there exists at least one data point where the 90-second window heart rate matrix profile
    exceeds 7.3 within a 90-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    hr_mp = get_vitals(x.pt_id, 
                       "HR MP90",
                       t_start=(x.alarm_datetime - timedelta(seconds=45)),
                       t_end=(x.alarm_datetime + timedelta(seconds=45)) )
    return SUPPRESSIBLE if np.any( hr_mp > 7.3 ) else ABSTAIN


@labeling_function()
def lf_outlier_hr_80(x):
    """
    If there exists at least one data point where the 80-second window heart rate matrix profile
    exceeds 6.7 within a 80-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    hr_mp = get_vitals(x.pt_id, 
                       "HR MP80",
                       t_start=(x.alarm_datetime - timedelta(seconds=40)),
                       t_end=(x.alarm_datetime + timedelta(seconds=40)) )
    return SUPPRESSIBLE if np.any( hr_mp > 6.7 ) else ABSTAIN


@labeling_function()
def lf_outlier_hr_70(x):
    """
    If there exists at least one data point where the 70-second window heart rate matrix profile
    exceeds 6.0 within a 70-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    hr_mp = get_vitals(x.pt_id, 
                       "HR MP70",
                       t_start=(x.alarm_datetime - timedelta(seconds=35)),
                       t_end=(x.alarm_datetime + timedelta(seconds=35)) )
    return SUPPRESSIBLE if np.any( hr_mp > 6.0 ) else ABSTAIN


@labeling_function()
def lf_outlier_hr_60(x):
    """
    If there exists at least one data point where the 60-second window heart rate matrix profile
    exceeds 5.4 within a 60-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    hr_mp = get_vitals(x.pt_id, 
                       "HR MP60",
                       t_start=(x.alarm_datetime - timedelta(seconds=30)),
                       t_end=(x.alarm_datetime + timedelta(seconds=30)) )
    return SUPPRESSIBLE if np.any( hr_mp > 5.4 ) else ABSTAIN


@labeling_function()
def lf_outlier_hr_50(x):
    """
    If there exists at least one data point where the 50-second window heart rate matrix profile
    exceeds 4.7 within a 50-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    hr_mp = get_vitals(x.pt_id, 
                       "HR MP50",
                       t_start=(x.alarm_datetime - timedelta(seconds=25)),
                       t_end=(x.alarm_datetime + timedelta(seconds=25)) )
    return SUPPRESSIBLE if np.any( hr_mp > 4.7 ) else ABSTAIN


@labeling_function()
def lf_outlier_hr_40(x):
    """
    If there exists at least one data point where the 40-second window heart rate matrix profile
    exceeds 3.9 within a 40-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    hr_mp = get_vitals(x.pt_id, 
                       "HR MP40",
                       t_start=(x.alarm_datetime - timedelta(seconds=20)),
                       t_end=(x.alarm_datetime + timedelta(seconds=20)) )
    return SUPPRESSIBLE if np.any( hr_mp > 3.9 ) else ABSTAIN


@labeling_function()
def lf_outlier_hr_30(x):
    """
    If there exists at least one data point where the 30-second window heart rate matrix profile
    exceeds 3.1 within a 30-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    hr_mp = get_vitals(x.pt_id, 
                       "HR MP30",
                       t_start=(x.alarm_datetime - timedelta(seconds=15)),
                       t_end=(x.alarm_datetime + timedelta(seconds=15)) )
    return SUPPRESSIBLE if np.any( hr_mp > 3.1 ) else ABSTAIN


@labeling_function()
def lf_outlier_hr_20(x):
    """
    If there exists at least one data point where the 20-second window heart rate matrix profile
    exceeds 2.1 within a 20-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    hr_mp = get_vitals(x.pt_id, 
                       "HR MP20",
                       t_start=(x.alarm_datetime - timedelta(seconds=10)),
                       t_end=(x.alarm_datetime + timedelta(seconds=10)) )
    return SUPPRESSIBLE if np.any( hr_mp > 2.1 ) else ABSTAIN


@labeling_function()
def lf_outlier_rr_120(x):
    """
    If there exists at least one data point where the 120-second window respiratory rate matrix profile
    exceeds 8.7 within a 120-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    rr_mp = get_vitals(x.pt_id, 
                       "RESP MP120",
                       t_start=(x.alarm_datetime - timedelta(seconds=60)),
                       t_end=(x.alarm_datetime + timedelta(seconds=60)) )
    return SUPPRESSIBLE if np.any( rr_mp > 8.7 ) else ABSTAIN


@labeling_function()
def lf_outlier_rr_110(x):
    """
    If there exists at least one data point where the 110-second window respiratory rate matrix profile
    exceeds 8.1 within a 110-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    rr_mp = get_vitals(x.pt_id, 
                       "RESP MP110",
                       t_start=(x.alarm_datetime - timedelta(seconds=55)),
                       t_end=(x.alarm_datetime + timedelta(seconds=55)))
    return SUPPRESSIBLE if np.any( rr_mp > 8.1 ) else ABSTAIN


@labeling_function()
def lf_outlier_rr_100(x):
    """
    If there exists at least one data point where the 100-second window respiratory rate matrix profile
    exceeds 7.6 within a 100-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    rr_mp = get_vitals(x.pt_id, 
                       "RESP MP100",
                       t_start=(x.alarm_datetime - timedelta(seconds=50)),
                       t_end=(x.alarm_datetime + timedelta(seconds=50)) )
    return SUPPRESSIBLE if np.any( rr_mp > 7.6 ) else ABSTAIN


@labeling_function()
def lf_outlier_rr_90(x):
    """
    If there exists at least one data point where the 90-second window respiratory rate matrix profile
    exceeds 7.1 within a 90-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    rr_mp = get_vitals(x.pt_id, 
                       "RESP MP90",
                       t_start=(x.alarm_datetime - timedelta(seconds=45)),
                       t_end=(x.alarm_datetime + timedelta(seconds=45)) )
    return SUPPRESSIBLE if np.any( rr_mp > 7.1 ) else ABSTAIN


@labeling_function()
def lf_outlier_rr_80(x):
    """
    If there exists at least one data point where the 80-second window respiratory rate matrix profile
    exceeds 6.5 within a 80-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    rr_mp = get_vitals(x.pt_id, 
                       "RESP MP80",
                       t_start=(x.alarm_datetime - timedelta(seconds=40)),
                       t_end=(x.alarm_datetime + timedelta(seconds=40)) )
    return SUPPRESSIBLE if np.any( rr_mp > 6.5 ) else ABSTAIN


@labeling_function()
def lf_outlier_rr_70(x):
    """
    If there exists at least one data point where the 70-second window respiratory rate matrix profile
    exceeds 6.0 within a 70-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    rr_mp = get_vitals(x.pt_id, 
                       "RESP MP70",
                       t_start=(x.alarm_datetime - timedelta(seconds=35)),
                       t_end=(x.alarm_datetime + timedelta(seconds=35)) )
    return SUPPRESSIBLE if np.any( rr_mp > 6.0 ) else ABSTAIN


@labeling_function()
def lf_outlier_rr_60(x):
    """
    If there exists at least one data point where the 60-second window respiratory rate matrix profile
    exceeds 5.4 within a 60-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    rr_mp = get_vitals(x.pt_id, 
                       "RESP MP60",
                       t_start=(x.alarm_datetime - timedelta(seconds=30)),
                       t_end=(x.alarm_datetime + timedelta(seconds=30)) )
    return SUPPRESSIBLE if np.any( rr_mp > 5.4 ) else ABSTAIN


@labeling_function()
def lf_outlier_rr_50(x):
    """
    If there exists at least one data point where the 50-second window respiratory rate matrix profile
    exceeds 4.7 within a 50-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    rr_mp = get_vitals(x.pt_id, 
                       "RESP MP50",
                       t_start=(x.alarm_datetime - timedelta(seconds=25)),
                       t_end=(x.alarm_datetime + timedelta(seconds=25)) )
    return SUPPRESSIBLE if np.any( rr_mp > 4.7 ) else ABSTAIN


@labeling_function()
def lf_outlier_rr_40(x):
    """
    If there exists at least one data point where the 40-second window respiratory rate matrix profile
    exceeds 3.9 within a 40-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    rr_mp = get_vitals(x.pt_id, 
                       "RESP MP40",
                       t_start=(x.alarm_datetime - timedelta(seconds=20)),
                       t_end=(x.alarm_datetime + timedelta(seconds=20)) )
    return SUPPRESSIBLE if np.any( rr_mp > 3.9 ) else ABSTAIN


@labeling_function()
def lf_outlier_rr_30(x):
    """
    If there exists at least one data point where the 30-second window respiratory rate matrix profile
    exceeds 3.0 within a 30-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    rr_mp = get_vitals(x.pt_id, 
                       "RESP MP30",
                       t_start=(x.alarm_datetime - timedelta(seconds=15)),
                       t_end=(x.alarm_datetime + timedelta(seconds=15)) )
    return SUPPRESSIBLE if np.any( rr_mp > 3.0 ) else ABSTAIN


@labeling_function()
def lf_outlier_rr_20(x):
    """
    If there exists at least one data point where the 20-second window respiratory rate matrix profile
    exceeds 2.0 within a 20-second window surrounding the alarm then the alarm is suppressible, 
    otherwise abstain
    """
    rr_mp = get_vitals(x.pt_id, 
                       "RESP MP20",
                       t_start=(x.alarm_datetime - timedelta(seconds=10)),
                       t_end=(x.alarm_datetime + timedelta(seconds=10)) )
    return SUPPRESSIBLE if np.any( rr_mp > 2.0 ) else ABSTAIN


# ## Apply Labeling Functions to Data

# In[ ]:


lfs = [
        lf_long_alarm_60s, lf_long_alarm_65s, lf_long_alarm_70s,
        lf_spo2_below85_over120s, lf_spo2_below80_over100s, lf_spo2_below70_over90s, lf_spo2_below60_over60s, lf_spo2_below50_over30s,
        lf_hr_below50_over120s, lf_hr_below40_over60s, lf_hr_below30,
        lf_rr_below50_over120s, lf_rr_below40_over60s, lf_rr_below30,
        lf_repeat_alarms_15s, lf_repeat_alarms_30s, lf_repeat_alarms_60s,
        lf_short_alarm_15s, lf_short_alarm_10s, lf_short_alarm_5s,
        lf_immediate_recovery_10s, lf_immediate_recovery_15s,
        lf_hr_tech_err_20, lf_hr_tech_err_30,
        lf_outlier_spo2_120, lf_outlier_spo2_110, lf_outlier_spo2_100, lf_outlier_spo2_90, lf_outlier_spo2_80, lf_outlier_spo2_70, lf_outlier_spo2_60, lf_outlier_spo2_50, lf_outlier_spo2_40, lf_outlier_spo2_30, lf_outlier_spo2_20,
        lf_outlier_hr_120, lf_outlier_hr_110, lf_outlier_hr_100, lf_outlier_hr_90, lf_outlier_hr_80, lf_outlier_hr_70, lf_outlier_hr_60, lf_outlier_hr_50, lf_outlier_hr_40, lf_outlier_hr_30, lf_outlier_hr_20,
        lf_outlier_rr_120, lf_outlier_rr_110, lf_outlier_rr_100, lf_outlier_rr_90, lf_outlier_rr_80, lf_outlier_rr_70, lf_outlier_rr_60, lf_outlier_rr_50, lf_outlier_rr_40, lf_outlier_rr_30, lf_outlier_rr_20,
        lf_spo2_below80_over120s, lf_hr_above220_over10s, lf_hr_below50_over10s, lf_rr_below10_over60s, lf_10alarms_5mins
    ]


# Note: LF application takes approx 25 minutes on Ash, if you have "data/L_alarms_62.npy" then load that instead (2 cells down)

# In[ ]:


#get_ipython().run_cell_magic('time', '', 'applier = PandasParallelLFApplier(lfs)\nL_alarms = applier.apply(alarms_df, n_parallel=10, scheduler="threads", fault_tolerant=True)\nY_alarms = alarms_df.true_label.values')


# In[ ]:


#np.save("./L_alarms.npy", L_alarms)
L_data = np.load("/data2/sfpugh/smart-alarm-2.0/chop-alarm-data/L_alarms_62.npy")
Y_data = alarms_df.true_label.values


# ### LFAnalysis Summary Statistics:
# - Polarity - the set of unique labels this LF outputs
# - Coverage - the fraction of the dataset this LF labels
# - Overlaps - the fraction of the dataset where this LF and at least one other LF label
# - Conflicts - the fraction of the dataset where this LF and at least on other LF label and disagree
# - Correct - the number of data points this LF labels correctly
# - Incorrrect - the number of data points this LF labels incorrectly
# - Emp. Acc. - empirical accuracy of LF (on data points with non-abstain labels only)
# - Learned Weight - learned LF weight for combining LFs

# In[ ]:


LFAnalysis(L_data, lfs).lf_summary(Y=Y_data)