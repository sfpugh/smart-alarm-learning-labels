from snorkel.labeling import labeling_function
from snorkel.preprocess import preprocessor
import pandas as pd
import numpy as np
import wfdb
from matrixprofile import matrixProfile
import matplotlib.pyplot as plt
import time

MIMIC3WDB = 'mimic3wdb/matched/'
ABSTAIN = -1
NOT_SUPPRESSIBLE = 0
SUPPRESSIBLE = 1

np.seterr(divide='ignore', invalid='ignore')

"""The distance profile is computed by mass() and massStomp() subroutine, which normalize the profile value
    by dividing standard deviation of the moving window, causing error when the window has std = 0.
    To avoid normalization, please add this line to mass() and massStomp() in matrixprofile.utils.py:
    
    std[std == 0] = 1
    
    and add this to mass():
    
    if q_std == 0:
        q_std = 1
"""


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


# Same preprocessor retrieving SpO2 waveform
@preprocessor()
def retrieve_spo2_waveform(x):
    rec_path = x.mimic3wdb_matched_ref.split('/')
    pb_dir = MIMIC3WDB + rec_path[0] + '/' + rec_path[1]

    try:
        record = wfdb.rdrecord(rec_path[2], channel_names=['SpO2'], pb_dir=pb_dir)
    except:
        record = wfdb.rdrecord(rec_path[2], channel_names=['%SpO2'], pb_dir=pb_dir)

    x.spo2 = record.p_signal[:, 0]

    return x


# Same preprocessor retrieving HR waveform
@preprocessor()
def retrieve_hr_waveform(x):
    rec_path = x.mimic3wdb_matched_ref.split('/')
    pb_dir = MIMIC3WDB + rec_path[0] + '/' + rec_path[1]

    record = wfdb.rdrecord(rec_path[2], channel_names=['HR'], pb_dir=pb_dir)
    x.hr = record.p_signal[:, 0]

    return x


# Same preprocessor retrieving RR waveform
@preprocessor()
def retrieve_rr_waveform(x):
    rec_path = x.mimic3wdb_matched_ref.split('/')
    pb_dir = MIMIC3WDB + rec_path[0] + '/' + rec_path[1]

    record = wfdb.rdrecord(rec_path[2], channel_names=['RESP'], pb_dir=pb_dir)
    x.resp = record.p_signal[:, 0]

    return x


# All labeling functions have thresholds tuned such that %outlier = 5%
# SpO2 outlier detection
@labeling_function(pre=[retrieve_spo2_waveform])
def lf_outlier_spo2_120(x):
    x.spo2_mp, _ = compute_mp(x.spo2, window=120, threshold=8.4)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.spo2_mp, length=120, center=x.alarm_start) > 8.4) else ABSTAIN


@labeling_function(pre=[retrieve_spo2_waveform])
def lf_outlier_spo2_110(x):
    x.spo2_mp, _ = compute_mp(x.spo2, window=110, threshold=7.8)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.spo2_mp, length=110, center=x.alarm_start) > 7.8) else ABSTAIN


@labeling_function(pre=[retrieve_spo2_waveform])
def lf_outlier_spo2_100(x):
    x.spo2_mp, _ = compute_mp(x.spo2, window=100, threshold=7.2)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.spo2_mp, length=100, center=x.alarm_start) > 7.2) else ABSTAIN


@labeling_function(pre=[retrieve_spo2_waveform])
def lf_outlier_spo2_90(x):
    x.spo2_mp, _ = compute_mp(x.spo2, window=90, threshold=6.6)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.spo2_mp, length=90, center=x.alarm_start) > 6.6) else ABSTAIN


@labeling_function(pre=[retrieve_spo2_waveform])
def lf_outlier_spo2_80(x):
    x.spo2_mp, _ = compute_mp(x.spo2, window=80, threshold=6)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.spo2_mp, length=80, center=x.alarm_start) > 6) else ABSTAIN


@labeling_function(pre=[retrieve_spo2_waveform])
def lf_outlier_spo2_70(x):
    x.spo2_mp, _ = compute_mp(x.spo2, window=70, threshold=5.3)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.spo2_mp, length=70, center=x.alarm_start) > 5.3) else ABSTAIN


@labeling_function(pre=[retrieve_spo2_waveform])
def lf_outlier_spo2_60(x):
    x.spo2_mp, _ = compute_mp(x.spo2, window=60, threshold=4.6)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.spo2_mp, length=60, center=x.alarm_start) > 4.6) else ABSTAIN


@labeling_function(pre=[retrieve_spo2_waveform])
def lf_outlier_spo2_50(x):
    x.spo2_mp, _ = compute_mp(x.spo2, window=50, threshold=3.8)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.spo2_mp, length=50, center=x.alarm_start) > 3.8) else ABSTAIN


@labeling_function(pre=[retrieve_spo2_waveform])
def lf_outlier_spo2_40(x):
    x.spo2_mp, _ = compute_mp(x.spo2, window=40, threshold=2.9)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.spo2_mp, length=40, center=x.alarm_start) > 2.9) else ABSTAIN


@labeling_function(pre=[retrieve_spo2_waveform])
def lf_outlier_spo2_30(x):
    x.spo2_mp, _ = compute_mp(x.spo2, window=30, threshold=2.1)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.spo2_mp, length=30, center=x.alarm_start) > 2.1) else ABSTAIN


@labeling_function(pre=[retrieve_spo2_waveform])
def lf_outlier_spo2_20(x):
    x.spo2_mp, _ = compute_mp(x.spo2, window=20, threshold=1)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.spo2_mp, length=20, center=x.alarm_start) > 1) else ABSTAIN


# one function with very long window
@labeling_function(pre=[retrieve_spo2_waveform])
def lf_outlier_spo2_3000(x):
    x.spo2_mp, _ = compute_mp(x.spo2, window=3000, threshold=72)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.spo2_mp, length=3000, center=x.alarm_start) > 72) else ABSTAIN


# HR outlier detection
@labeling_function(pre=[retrieve_hr_waveform])
def lf_outlier_hr_120(x):
    x.hr_mp, _ = compute_mp(x.hr, window=120, threshold=9)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.hr_mp, length=120, center=x.alarm_start) > 9) else ABSTAIN


@labeling_function(pre=[retrieve_hr_waveform])
def lf_outlier_hr_110(x):
    x.hr_mp, _ = compute_mp(x.hr, window=110, threshold=8.5)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.hr_mp, length=110, center=x.alarm_start) > 8.5) else ABSTAIN


@labeling_function(pre=[retrieve_hr_waveform])
def lf_outlier_hr_100(x):
    x.hr_mp, _ = compute_mp(x.hr, window=100, threshold=7.8)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.hr_mp, length=100, center=x.alarm_start) > 7.8) else ABSTAIN


@labeling_function(pre=[retrieve_hr_waveform])
def lf_outlier_hr_90(x):
    x.hr_mp, _ = compute_mp(x.hr, window=90, threshold=7.3)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.hr_mp, length=90, center=x.alarm_start) > 7.3) else ABSTAIN


@labeling_function(pre=[retrieve_hr_waveform])
def lf_outlier_hr_80(x):
    x.hr_mp, _ = compute_mp(x.hr, window=80, threshold=6.7)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.hr_mp, length=80, center=x.alarm_start) > 6.7) else ABSTAIN


@labeling_function(pre=[retrieve_hr_waveform])
def lf_outlier_hr_70(x):
    x.hr_mp, _ = compute_mp(x.hr, window=70, threshold=6)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.hr_mp, length=70, center=x.alarm_start) > 6) else ABSTAIN


@labeling_function(pre=[retrieve_hr_waveform])
def lf_outlier_hr_60(x):
    x.hr_mp, _ = compute_mp(x.hr, window=60, threshold=5.4)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.hr_mp, length=60, center=x.alarm_start) > 5.4) else ABSTAIN


@labeling_function(pre=[retrieve_hr_waveform])
def lf_outlier_hr_50(x):
    x.hr_mp, _ = compute_mp(x.hr, window=50, threshold=4.7)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.hr_mp, length=50, center=x.alarm_start) > 4.7) else ABSTAIN


@labeling_function(pre=[retrieve_hr_waveform])
def lf_outlier_hr_40(x):
    x.hr_mp, _ = compute_mp(x.hr, window=40, threshold=3.9)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.hr_mp, length=40, center=x.alarm_start) > 3.9) else ABSTAIN


@labeling_function(pre=[retrieve_hr_waveform])
def lf_outlier_hr_30(x):
    x.hr_mp, _ = compute_mp(x.hr, window=30, threshold=3.1)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.hr_mp, length=30, center=x.alarm_start) > 3.1) else ABSTAIN


@labeling_function(pre=[retrieve_hr_waveform])
def lf_outlier_hr_20(x):
    x.hr_mp, _ = compute_mp(x.hr, window=20, threshold=2.1)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.hr_mp, length=20, center=x.alarm_start) > 2.1) else ABSTAIN


# RR outlier detection
@labeling_function(pre=[retrieve_rr_waveform])
def lf_outlier_resp_120(x):
    x.resp_mp, _ = compute_mp(x.resp, window=120, threshold=8.7)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.resp_mp, length=120, center=x.alarm_start) > 8.7) else ABSTAIN


@labeling_function(pre=[retrieve_rr_waveform])
def lf_outlier_resp_110(x):
    x.resp_mp, _ = compute_mp(x.resp, window=110, threshold=8.1)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.resp_mp, length=110, center=x.alarm_start) > 8.1) else ABSTAIN


@labeling_function(pre=[retrieve_rr_waveform])
def lf_outlier_resp_100(x):
    x.resp_mp, _ = compute_mp(x.resp, window=100, threshold=7.6)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.resp_mp, length=100, center=x.alarm_start) > 7.6) else ABSTAIN


@labeling_function(pre=[retrieve_rr_waveform])
def lf_outlier_resp_90(x):
    x.resp_mp, _ = compute_mp(x.resp, window=90, threshold=7.1)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.resp_mp, length=90, center=x.alarm_start) > 7.1) else ABSTAIN


@labeling_function(pre=[retrieve_rr_waveform])
def lf_outlier_resp_80(x):
    x.resp_mp, _ = compute_mp(x.resp, window=80, threshold=6.5)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.resp_mp, length=80, center=x.alarm_start) > 6.5) else ABSTAIN


@labeling_function(pre=[retrieve_rr_waveform])
def lf_outlier_resp_70(x):
    x.resp_mp, _ = compute_mp(x.resp, window=70, threshold=6)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.resp_mp, length=70, center=x.alarm_start) > 6) else ABSTAIN


@labeling_function(pre=[retrieve_rr_waveform])
def lf_outlier_resp_60(x):
    x.resp_mp, _ = compute_mp(x.resp, window=60, threshold=5.4)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.resp_mp, length=60, center=x.alarm_start) > 5.4) else ABSTAIN


@labeling_function(pre=[retrieve_rr_waveform])
def lf_outlier_resp_50(x):
    x.resp_mp, _ = compute_mp(x.resp, window=50, threshold=4.7)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.resp_mp, length=50, center=x.alarm_start) > 4.7) else ABSTAIN


@labeling_function(pre=[retrieve_rr_waveform])
def lf_outlier_resp_40(x):
    x.resp_mp, _ = compute_mp(x.resp, window=40, threshold=3.9)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.resp_mp, length=40, center=x.alarm_start) > 3.9) else ABSTAIN


@labeling_function(pre=[retrieve_rr_waveform])
def lf_outlier_resp_30(x):
    x.resp_mp, _ = compute_mp(x.resp, window=30, threshold=3)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.resp_mp, length=30, center=x.alarm_start) > 3) else ABSTAIN


@labeling_function(pre=[retrieve_rr_waveform])
def lf_outlier_resp_20(x):
    x.resp_mp, _ = compute_mp(x.resp, window=20, threshold=2)
    return SUPPRESSIBLE if np.any(interval_centered_at(x.resp_mp, length=20, center=x.alarm_start) > 2) else ABSTAIN


# test the implementation on one spo2/hr/resp time series
if __name__ == "__main__":
    w = 20
    t = 2
    mimiciii_spo2alarms = pd.read_pickle('./spo2_alarms_df_v4.20.pkl')
    x = mimiciii_spo2alarms.iloc[0]
    x = retrieve_rr_waveform(x)

    # Compute matrix profile
    x.resp_mp, outlier_percent = compute_mp(x.resp, window=w, threshold=t)
    s = time.time()
    print('elapsed time: ' + str(time.time() - s) + 's')
    print("window = " + str(w) + " threshold = " + str(t) + " len = " + str(len(x.resp)) + " %outlier = " + str(
        outlier_percent))

    # Plot the signal data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    ax1.plot(np.arange(len(x.resp)), x.resp, label="SpO2 Data")
    ax1.set_ylabel('Signal', size=22)

    # Plot the Matrix Profile
    ax2.plot(range(15))
    ax2.plot(np.arange(len(x.resp_mp)), [t] * len(x.resp_mp), color='black')
    ax2.plot(np.arange(len(x.resp_mp)), x.resp_mp, label="Matrix Profile", color='red')
    ax2.set_ylabel('Matrix Profile', size=22)

    plt.show()