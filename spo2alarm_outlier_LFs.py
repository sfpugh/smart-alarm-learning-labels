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

MP_WINDOW = 120 # matrix profile window size
MP_THRESHOLD = 10  # matrix profile threshold, decide outlier if > threshold

np.seterr(divide='ignore', invalid='ignore')

"""The distance profile is computed by mass() and massStomp() subroutine, which normalize the profile value
    by dividing standard deviation of the moving window, causing error when the window has std = 0.
    To avoid normalization, please edit mass() in matrixprofile.utils.py to:
    
    def mass(query,ts, normalize=False):
        #query_normalized = zNormalize(np.copy(query))
        m = len(query)
        q_mean = np.mean(query)
        q_std = np.std(query)
        mean, std = movmeanstd(ts,m)
        dot = slidingDotProduct(query,ts)
        #res = np.sqrt(2*m*(1-(dot-m*mean*q_mean)/(m*std*q_std)))
        if normalize:
            res = 2*m*(1-(dot-m*mean*q_mean)/(m*std*q_std))
        else:
            res = 2*m*(1-(dot-m*mean*q_mean)/m)
    
        return res
        
    And do the same to massStomp()
"""


@preprocessor()
def retrieve_spo2_mp(x):
    rec_path = x.mimic3wdb_matched_ref.split('/')
    pb_dir = MIMIC3WDB + rec_path[0] + '/' + rec_path[1]

    try:
        record = wfdb.rdrecord(rec_path[2], channel_names=['SpO2'], pb_dir=pb_dir)
    except:
        record = wfdb.rdrecord(rec_path[2], channel_names=['%SpO2'], pb_dir=pb_dir)

    x.spo2 = record.p_signal[:, 0]
    x.spo2_mp = matrixProfile.stomp(x.spo2, m=MP_WINDOW)
    x.spo2_mp = np.nan_to_num(x.spo2_mp)
    return x


@labeling_function(pre=[retrieve_spo2_mp])
def lf_outlier_120(x):
    return NOT_SUPPRESSIBLE if np.any(x.spo2_mp[x.alarm_start - 120: x.alarm_start + 120] > MP_THRESHOLD) else ABSTAIN


@labeling_function(pre=[retrieve_spo2_mp])
def lf_outlier_100(x):
    return SUPPRESSIBLE if np.any(x.spo2_mp[x.alarm_start - 100: x.alarm_start + 100] > MP_THRESHOLD) else ABSTAIN


@labeling_function(pre=[retrieve_spo2_mp])
def lf_outlier_90(x):
    return SUPPRESSIBLE if np.any(x.spo2_mp[x.alarm_start - 90: x.alarm_start + 90] > MP_THRESHOLD) else ABSTAIN


@labeling_function(pre=[retrieve_spo2_mp])
def lf_outlier_60(x):
    return SUPPRESSIBLE if np.any(x.spo2_mp[x.alarm_start - 60: x.alarm_start + 60] > MP_THRESHOLD) else ABSTAIN


@labeling_function(pre=[retrieve_spo2_mp])
def lf_outlier_30(x):
    return SUPPRESSIBLE if np.any(x.spo2_mp[x.alarm_start - 30: x.alarm_start + 30] > MP_THRESHOLD) else ABSTAIN


# testing on one time series to see if matrix profile works
if __name__ == "__main__":
    mimiciii_spo2alarms = pd.read_pickle('./spo2_alarms_df_v4.20.pkl')
    x = mimiciii_spo2alarms.iloc[0]
    s = time.time()
    x = retrieve_spo2_mp(x)
    print(x.spo2)
    print(x.spo2_mp[0])
    print('elapsed time: ' + str(time.time() - s) + 's')

    # Append np.nan to Matrix profile to enable plotting against raw data
    mp_adj = np.append(x.spo2_mp[0], np.zeros(MP_WINDOW - 1) + np.nan)

    # Plot the signal data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    ax1.plot(np.arange(len(x.spo2)), x.spo2, label="SpO2 Data")
    ax1.set_ylabel('Signal', size=22)

    # Plot the Matrix Profile
    ax2.plot(np.arange(len(mp_adj)), mp_adj, label="Matrix Profile", color='red')
    ax2.set_ylabel('Matrix Profile', size=22)

    plt.show()