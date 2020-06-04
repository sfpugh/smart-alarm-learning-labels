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
                                    'hr_age_factor':[1,1,1,1], 
                                    'rr_age_factor':[1,1,1,1]}, 
                                    index=[1,2,3,4]) 


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
    idxs = spo2.index
    print(idxs)
    print(idxs[1])

    return max(r)


@labeling_function()
def lf_immediate_recovery_10s(x):
    """
    If SpO2 level increases/recovers by more than 20 percentage points within 
    10 seconds of alarm start then the alarm is suppressible, otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    spo2_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=10), ['SPO2-%']]
    return SUPPRESSIBLE if max_recovery(spo2_subrange) > 20 else ABSTAIN


@labeling_function()
def lf_immediate_recovery_15s(x):
    """
    If SpO2 level increases/recovers by more than 30 percentage points within 
    15 seconds of alarm start then the alarm is suppressible, otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    spo2_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=15), ['SPO2-%']]
    return SUPPRESSIBLE if max_recovery(spo2_subrange) > 30 else ABSTAIN


@labeling_function()
def lf_hr_tech_err_20(x):
    """
    If the difference between the SpO2 HR and ECG HR is larger than 20 percentage points 
    at time of alarm then suppressible, otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    t = v_df.index.get_loc(x.alarm_datetime, method='bfill')  # or use 'nearest' to get closest timestamp in general
    return SUPPRESSIBlE if abs(v_df.at[t,'SPO2-R'] - v_df.at[t,'HR']) > 20 else ABSTAIN


@labeling_function()
def lf_hr_tech_err_30(x):
    """
    If the difference between the SpO2 HR and ECG HR is larger than 30 percentage points 
    at time of alarm then suppressible, otherwise abstain
    """
    v_df = vitals_dfs[x.pt_id]
    t = v_df.index.get_loc(x.alarm_datetime, method='bfill')  # or use 'nearest' to get closest timestamp in general
    return SUPPRESSIBlE if abs(v_df.at[t,'SPO2-R'] - v_df.at[t,'HR']) > 30 else ABSTAIN


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


# retrieve interval centered at some time stamp
def interval_centered_at(x, length, center):
    start = max(0, center - length//2)
    end = min(len(x)-1, center + length//2)
    return x[start: end]


# compute matrix profile at given window and threshold, return mp and percentage of outliers
def compute_mp(ts, window, threshold):
    # remove trailing nans of ts
    i = len(ts) - 1
    while np.isnan(ts[i]) and i >= 0:
        i -= 1
    ts = ts[0:i]
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
    # compute percentage of outliers, where head, tail and zero std points do not participate
    outlier = mp[np.where(mp > threshold)]
    outlier_percentage = len(outlier) / (len(mp) - len(mp_head) - len(mp_tail) - count_zero_std)
    return mp, outlier_percentage


@labeling_function()
def lf_outlier_spo2_110(x):
    v_df = vitals_dfs[x.pt_id]
    spo2_mp, _ = compute_mp(v_df['SPO2-%'], window=110, threshold=7.8)
    return SUPPRESSIBLE if np.any(interval_centered_at(spo2_mp, length=110, center=x.alarm_datetime) > 7.8) else ABSTAIN


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
    ## Apply LFs to all alarms
    lfs = [lf_short_alarm_15s, lf_short_alarm_10s, lf_short_alarm_5s,
            lf_long_alarm_60s, lf_long_alarm_65s, lf_long_alarm_70s,
            lf_hr_tech_err_20, lf_hr_tech_err_30,
            lf_spo2_below85_over120s, lf_spo2_below80_over100s, lf_spo2_below70_over90s, lf_spo2_below60_over60s, lf_spo2_below50_over30s,
            lf_hr_below50_over120s, lf_hr_below40_over60s, lf_hr_below30,
            lf_rr_below50_over120s, lf_rr_below40_over60s, lf_rr_below30,
            lf_repeat_alarms_15s, lf_repeat_alarms_30s, lf_repeat_alarms_60s]

    print('Applying LFs...\n')
    applier = PandasParallelLFApplier(lfs)
    L_train = applier.apply(alarms_df, n_parallel=10, scheduler='threads', fault_tolerant=True)
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

    lf_outlier_spo2_110(x)
    
    exit(0)


if __name__ == '__main__':
    #main()
    test_lf()
