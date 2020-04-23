from google.oauth2 import service_account
import pandas as pd
import numpy as np
import datetime
import time
import sys
import wfdb

MIMIC3WDB = 'mimic3wdb/matched/'
VALUENUM = 4
SPO2_LO = 5820
SPO2_HI = 8554

s = time.time()

## set display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=sys.maxsize)


## Read SpO2 alarm settings from BigQuery table
credentials = service_account.Credentials.from_service_account_file('vibrant-sound-266816-3e9c5c8b4681.json')

alarm_settings_df = pd.read_gbq('''
    SELECT SUBJECT_ID, ITEMID, CHARTTIME, VALUENUM
    FROM `vibrant-sound-266816.project_tables.spo2_alarm_settings`
    WHERE ITEMID = 5820
    ORDER BY SUBJECT_ID
''', project_id=credentials.project_id, credentials=credentials)

# fix some incorrect settings
alarm_settings_df.loc[alarm_settings_df.VALUENUM == 9, 'VALUENUM'] = 90
alarm_settings_df.loc[alarm_settings_df.VALUENUM == 8, 'VALUENUM'] = 80
alarm_settings_df.loc[alarm_settings_df.VALUENUM == 7, 'VALUENUM'] = 70
alarm_settings_df.loc[alarm_settings_df.VALUENUM == 6, 'VALUENUM'] = 60


## Extract the SpO2 alarm data
spo2_alarms_df = pd.DataFrame(columns=['alarm_start','alarm_end','mimic3wdb_matched_ref'])

with open("RECORDS-numerics", "r") as rec_file:
    for rec in rec_file:
        rec = rec.strip()
        rec_path = rec.split('/')

        # ignore records for which we do not have spo2 alarm settings
        subject_id = int(rec_path[2].split('-')[0].strip('p'))
        if not subject_id in alarm_settings_df.SUBJECT_ID.values:
            continue;
        
        try:
            # read spo2 waveform for record from mimic3wdb/matched
            pb_dir = MIMIC3WDB + rec_path[0] + '/' + rec_path[1]
            
            try:    # spo2 can be under two names, so catching cases where labeled differently
                record = wfdb.rdrecord(rec_path[2], channel_names=['SpO2'], pb_dir=pb_dir)
                spo2 = record.p_signal[:,0]
            except:
                record = wfdb.rdrecord(rec_path[2], channel_names=['%SpO2'], pb_dir=pb_dir)
                spo2 = record.p_signal[:,0]
            # if spo2 waveform dne then exception will be thrown and this record will be skipped


            # extract spo2 alarm settings per second for current subject
            temp_df = alarm_settings_df[alarm_settings_df.SUBJECT_ID == subject_id]
            lo_settings_df = pd.DataFrame(index=range(record.sig_len), columns=['SETTING'])

            for i, row in temp_df.iterrows():
                x = int( (row.CHARTTIME - record.base_datetime).total_seconds() )
                if x < 0:
                    lo_settings_df.iloc[0,0] = row.VALUENUM
                elif x < record.sig_len:
                    lo_settings_df.iloc[x,0] = row.VALUENUM
                # else >= sig_len and not in scope of this record

            lo_settings_df = lo_settings_df.fillna(method='ffill')    # forward fill NaNs

            # extract alarms
            spo2_alarmON = np.argwhere(spo2 < lo_settings_df.SETTING.to_numpy())[:,0]

            start = None
            for i, t in enumerate(spo2_alarmON):
                if start == None:
                    start = t
                elif t - spo2_alarmON[i-1] != 1:
                    spo2_alarms_df = spo2_alarms_df.append({'alarm_start':start, 'alarm_end':spo2_alarmON[i-1], 'mimic3wdb_matched_ref':rec}, ignore_index=True)
                    start = t
                elif i+1 == len(spo2_alarmON):  # for last alarm
                    spo2_alarms_df = spo2_alarms_df.append({'alarm_start':start, 'alarm_end':spo2_alarmON[i], 'mimic3wdb_matched_ref':rec}, ignore_index=True)


        except Exception as e:
            print('EXCEPTION occurred at line ' + str(sys.exc_info()[2].tb_lineno) + ' for record ' + rec)
            print('\t' + str(e) + '\n')

        
## export the spo2 alarm data
spo2_alarms_df.to_pickle('./smart-alarm-2.0/spo2_alarms_df.pkl')


print('elapsed time: ' + str(time.time() - s) + 's')
