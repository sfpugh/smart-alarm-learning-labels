import pandas as pd
import numpy as np
import zipfile
import datetime

DATA_DIR = './chop-alarm-data'
ALARMS_ZIP = 'zippedalarmfilesforPRECISE.zip'
VITALS_ZIP = 'zippedvitalsignfilesforPRECISE.zip'
LABELED_FILE_NAME = 'PRECISE alarm file labels 20161017.csv'
UNLABELED_FILE_NAME = 'PRECISE alarm file nolabels 20161017.csv'

TECHNICAL_ERR = 2

NOT_SUPPRESSIBLE = 0
SUPPRESSIBLE = 1

def read_alarms(save=False):
    """
    Reads alarms from zipped CSV and returns dataframe containing all spo2 low alarms
    """
    print('Processing alarms files...')

    zf = zipfile.ZipFile(DATA_DIR + '/' + ALARMS_ZIP)

    unlabeled_df = pd.read_csv( zf.open(UNLABELED_FILE_NAME), parse_dates=[['alarm_date1960','alarm_time']], keep_date_col=True )
    unlabeled_df = unlabeled_df.rename(columns={'alarm_date1960_alarm_time':'alarm_datetime'})
    unlabeled_df = unlabeled_df[ unlabeled_df['alarm_type'] == 'SPO LO' ]

    labeled_df = pd.read_csv( zf.open(LABELED_FILE_NAME) )
    labeled_df = labeled_df[ labeled_df['alarm_type'] == 'SPO LO' ]
    labeled_df = labeled_df.rename(columns={'monitor_alarm_cat':'monitor_alarm_cat_L','valid_final':'valid_final_L','pt_age_group':'pt_age_group_L'})

    df = pd.merge(unlabeled_df, labeled_df, on=['pt_id','alarm_date1960','alarm_time','alarm_type','duration'])
    df = df[['pt_id','alarm_date1960','alarm_time','alarm_datetime','monitor_alarm_cat','monitor_alarm_cat_L','alarm_type','duration','valid_final','valid_final_L','pt_age_group','pt_age_group_L']]

    df = df[ df['valid_final'] != TECHNICAL_ERR ]   # remove technical errors

    # generate suppressible/non-suppressible labels
    df['true_label'] = df['valid_final'].apply(lambda x : SUPPRESSIBLE if x == 0 else NOT_SUPPRESSIBLE)
    df['true_label_L'] = df['true_label'].apply(lambda x : 'Suppressible' if x == SUPPRESSIBLE else 'Not Suppressible')

    if save:
        df.to_pickle('spo2alarms_df.pkl')

    return df
# end read_alarms


def read_vitals(verbose=True, save=False):
    """
    Reads vitals from zipped CSVs and returns dictionary of vitals dataframes for each patient id
    """
    dfs = {}

    with zipfile.ZipFile(DATA_DIR + '/' + VITALS_ZIP) as zf:
        for vitals_file in zf.namelist():
            if int(vitals_file.split()[0]) in [580, 610]:
                print('Skipping ' + vitals_file + '...')
                continue

            if verbose:
                print('Processing ' + vitals_file + '...')
            
            # read excel file into dataframe
            df_original = pd.read_excel( zf.read(vitals_file) )

            # concatenate data so that have row per vital sign
            start = 0
            df = pd.DataFrame()
            for idx, row in df_original.iterrows():
                if np.all( row.isnull() ):
                    df_chunk = df_original[start:idx].reset_index(drop=True)
                    df = pd.concat([df, df_chunk], axis=1)
                    start = idx + 1
             
            df = pd.concat([df, df_original[start:].reset_index(drop=True)], axis=1)    # grab last chunk

            # reformat the dataframe
            df.iloc[0,0] = 'timestamp'

            if not int(vitals_file.split('.')[0].split()[0]) in [514,515,517]:   # these three files are formatted different
                df = df.T
                df = df.reset_index(drop=True)

            df.columns = df.iloc[0].values
            df = df.drop([0]).dropna(subset=['timestamp']).reset_index(drop=True)

            if int(vitals_file.split()[0]) == 610:
                print(df)

            df['datetime'] = np.nan
            base_date = datetime.date(1960,1,1) 
            for i, row in df.iterrows():
                if i-1 > 0 and df.at[i-1,'timestamp'] > row['timestamp']:
                    base_date = base_date + datetime.timedelta(days=1)
                if isinstance(row['timestamp'], str):
                    print(row['timestamp'])
                df.at[i,'datetime'] = datetime.datetime.combine(base_date, row['timestamp'])

            df = df.set_index('datetime', drop=True)

            # convert columns to numeric datatypes
            numeric_cols = [x for x in df.columns if x != 'timestamp']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])

            # add dataframe to dictionary
            dfs[int(vitals_file.split()[0])] = df

            if save:
                df.to_pickle(vitals_file.split('.')[0] + '.pkl')

    return dfs
# end read_vitals

