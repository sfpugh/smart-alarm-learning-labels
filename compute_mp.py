import numpy as np
import pandas as pd
from tqdm import tqdm
from matrixprofile import matrixProfile

tqdm.pandas()
np.seterr(divide='ignore', invalid='ignore')

def compute_std(ts, window):
    """
    Compute moving std
    
    Args:
        ts - time series data
        window - window (subsequence) length
    
    Return:
        float - moving std
    """
    ts_std = []
    for i in range(len(ts) - window):
        ts_std.append(np.std(ts[i: i+window-1]))
    ts_std = np.array(ts_std)
    ts_std_head = np.zeros(window // 2 - 1)
    ts_std_tail = np.zeros(len(ts) - len(ts_std) - window // 2 + 1)
    ts_std = np.concatenate([ts_std_head, ts_std, ts_std_tail])
    return ts_std


def compute_mp(ts, window, threshold=None):
    """
    Compute matrix profile at given window
    
    Args:
        ts - array containing time series data
        window - window length
        threshold - threshold for outlier value
    
    Return:
        numpy array - matrix profile
    """
    # sfp - commenting this out as it is unneccesary
    # remove trailing nans of ts
    #i = len(ts) - 1
    #while np.isnan(ts[i]) and i >= 0:
    #    i -= 1
    #ts = ts[0:i+1]
    
    # compute mp by stamp
    mp = np.array(matrixProfile.stomp(ts, m=window, ))[0]
    
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
    if not threshold is None:
        outlier = mp[np.where(mp > threshold)]
        outlier_percentage = len(outlier) / (len(mp) - len(mp_head) - len(mp_tail) - count_zero_std)
        print('outlier %: ' + str(outlier_percentage))
    
    return mp


def apply_compute_mp(df, vitals, verbose=True, save=False, filename=""):
    """
    Compute a matrix profile per patient with window sizes 20 to 120 seconds
    
    Args:
        df - dataframe containing vitals
        vitals - vital signs to compute matrix profile for
        verbose - True to print function time duration, otherwise False
        save - True to save dataframe with mp, otherwise False
        filename - Name of file to save dataframe to
        
    Return:
        dataframe - copy of vitals dataframe with new matrix profile columns
                    for SPO2, heart rate, and respiratory rate
    """
    start_time = time()
    
    new_df = df.copy()
    
    for vital in vitals:
        for w in range(20,121,10):
            mps = df.groupby('pt_id')[vital].progress_apply(lambda x: compute_mp(x.to_numpy(), w//INTERVAL))
            mps = mps.reindex(df.index.get_level_values(0).drop_duplicates())
            new_df[vital + ' MP' + str(w)] = mps.explode().to_numpy()

            # correctness check for 'apply_compute_mp'
            #for pt in new_df.index.levels[0]:
            #    if not np.all(new_df.loc[(pt, ), vital + ' MP' + str(w)].to_numpy() == mps[pt]):
            #        print('Non-match on patient ' + str(pt))
            
    if save:
        new_df.to_pickle(filename)
    
    if verbose:
        print(time() - start_time)
        
    return new_df