from google.oauth2 import service_account
import pandas as pd
import numpy as np
import datetime
import time
import sys
import wfdb

MIMIC3WDB = 'mimic3wdb/matched/'
VALUENUM = 5
SPO2_LO = 5820
SPO2_HI = 8554

start = time.time()

## set display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=sys.maxsize)

## Read SpO2 alarm settings from BigQuery table
credentials = service_account.Credentials.from_service_account_file('vibrant-sound-266816-3e9c5c8b4681.json')

alarm_settings_df = pd.read_gbq('''
    SELECT *
    FROM `vibrant-sound-266816.project_tables.pugh_spo2alarmsettings`
''', project_id=credentials.project_id, credentials=credentials)


## Extract the SpO2 alarm data
spo2_alarms_df = pd.DataFrame(columns=['alarm_start','alarm_end','mimic3wdb_matched_ref'])

with open("RECORDS-numerics", "r") as rec_file:
    for rec in rec_file:
        rec = rec.strip()
        rec_path = rec.split('/')

        # discard waveforms for which we do not have spo2 alarm settings
        subject_id = int(rec_path[2].split('-')[0].strip('p'))
        if not subject_id in alarm_settings_df.SUBJECT_ID.values:
            continue;
        
        try:
            # read spo2 waveform for record from mimic3wdb/matched
            pb_dir = MIMIC3WDB + rec_path[0] + '/' + rec_path[1]
            record = wfdb.rdrecord(rec_path[2], channel_names=['SpO2'], pb_dir=pb_dir)
            spo2 = record.p_signal[:,0]

            # extract spo2 alarm settings for subject
            lo_set = alarm_settings_df[(alarm_settings_df.SUBJECT_ID == subject_id) & (alarm_settings_df.ITEMID == SPO2_LO)].reset_index(drop=True)
            hi_set = alarm_settings_df[(alarm_settings_df.SUBJECT_ID == subject_id) & (alarm_settings_df.ITEMID == SPO2_HI)].reset_index(drop=True)

            # extract alarms
            spo2_alarmON = [i for (i,s) in enumerate(spo2) if s < lo_set.iloc[np.where(lo_set.CHARTTIME <= record.base_datetime + datetime.timedelta(seconds=i))[0].max(), VALUENUM]]

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

        
## export the spo2 alarm data to pickle file
spo2_alarms_df.to_pickle('./spo2_alarms_df.pkl')

print('elapsed time: ' + str(time.time() - start) + 's')
