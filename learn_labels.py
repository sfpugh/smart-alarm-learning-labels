from snorkel.labeling.apply.dask import PandasParallelLFApplier
from snorkel.labeling.model.label_model import LabelModel
from snorkel.labeling import labeling_function, LFAnalysis
from sklearn.metrics import confusion_matrix
from read_chop_data import read_alarms, read_vitals
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

## Global vars
# Label mappings
ABSTAIN = -1
NOT_SUPPRESSIBLE = 0
SUPPRESSIBLE = 1

AGE_FACTOR = 1

# Data dataframes
alarms_df = read_alarms()
vitals_dfs = read_vitals()


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


def max_recovery(x, spo2_data):
    r = []
    end = spo2_data.index[-1]
    print(end)

    for (i, row) in spo2_data.iterrows():
        print(i)

    return max(r)


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
def lf_repeat_alarms_5m(x):
    """
    If there exists other alarms within 5 minutes of the current alarm then the alarm is
    not suppressible, otherwise abstain
    """
    return repeat_alarms(x, 300) 


@labeling_function()
def lf_repeat_alarms_1m(x):
    """
    If there exists other alarms within 1 minute of the current alarm then the alarm is
    not suppressible, otherwise abstain
    """
    return repeat_alarms(x, 60) 


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
    #elif p[NOT_SUPPRESSIBLE] > p[SUPPRESSIBLE] or p[NOT_SUPPRESSIBLE] > 0.5:
        return NOT_SUPPRESSIBLE
    else:
        return ABSTAIN
# end pred_threshold


def main():
    ## Apply LFs to all alarms
    lfs = [lf_short_alarm_15s, lf_short_alarm_10s, lf_short_alarm_5s,
            lf_long_alarm_60s, lf_long_alarm_65s, lf_long_alarm_70s,
            lf_hr_tech_err_20, lf_hr_tech_err_30,
            lf_spo2_below85_over120s, lf_spo2_below80_over100s, lf_spo2_below70_over90s, lf_spo2_below60_over60s, lf_spo2_below50_over30s,
            lf_hr_below50_over120s, lf_hr_below40_over60s, lf_hr_below30,
            lf_rr_below50_over120s, lf_rr_below40_over60s, lf_rr_below30]
            #lf_repeat_alarms_5m, lf_repeat_alarms_1m]

    print('Applying LFs...\n')
    applier = PandasParallelLFApplier(lfs)
    L_train = applier.apply(alarms_df, n_parallel=10, scheduler='threads', fault_tolerant=True)
    print( LFAnalysis(L_train,lfs).lf_summary(), '\n' )


    ## Train a label model
    label_model = LabelModel(cardinality=2)
    label_model.fit(L_train)
    #label_model.save('label_model_' + datetime.now() + '.pkl')


    ## Evaluate label model accuracy

    # original prediction method
    pred = label_model.predict(L_train)
    print('Confusion Matrix (original): ')
    print( confusion_matrix(alarms_df.true_label, pred) )

    # predict with threshold
    pred_probs = label_model.predict_proba(L_train)

    for thres in [0.7,0.8,0.9]:
        pred = np.apply_along_axis(pred_threshold, 1, pred_probs, thres)
        print('Confusion Matrix (threshold=' + str(thres) + '):')
        print( confusion_matrix(alarms_df.true_label, pred) )
# end main


def test():
    x = alarms_df.iloc[194]
    print(x)

    v_df = vitals_dfs[x.pt_id]
    spo2_subrange = v_df.loc[x.alarm_datetime:x.alarm_datetime+timedelta(seconds=10), ['SPO2-%']]
    #print(spo2_subrange)

    max_recovery(x, spo2_subrange)
    
    exit(0)


if __name__ == '__main__':
    #test()
    main()
