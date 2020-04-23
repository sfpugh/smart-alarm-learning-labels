SUPPRESSIBLE = -1
UNKNOWN = 0
UNSUPPRESSIBLE = 1

# alarm_start_index[k], alarm_end_index[k] is the interval of the kth alarm


# 70 < SpO2 <= 85 for <= 60 sec
def spo2_70_85(alarm_start_index, alarm_end_index, spo2_waveform):
    labels = []
    n_alarms = len(alarm_start_index)
    current_alarm = 0
    for i in range(len(spo2_waveform)):
        if current_alarm >= n_alarms:
            labels.append(0)
            continue
        if i < alarm_start_index[current_alarm]:  # not an alarm
            labels.append(0)
        elif alarm_start_index[current_alarm] <= i <= alarm_end_index[current_alarm]:
            if spo2_waveform[i] <= 70 or spo2_waveform[i] > 85:
                labels.append(0)
                continue
            interval_start = max(0, i-59)
            interval_end = min(len(spo2_waveform), i+1)
            flag = 0
            for j in range(60):
                if all(70 < x <= 85 for x in spo2_waveform[interval_start: interval_end]):
                    labels.append(-1)
                    flag = 1
                    break
                else:
                    interval_start += 1
                    interval_end = min(len(spo2_waveform), interval_end + 1)
            if flag == 0:
                labels.append(0)
        elif i == alarm_end_index[current_alarm] + 1:  # not an alarm
            labels.append(0)
            current_alarm += 1
        else:  # never enter this case if data format is correct
            labels.append(0)
    return labels


# 60 < SpO2 <= 70 for <= 45 sec
def spo2_60_70(alarm_start_index, alarm_end_index, spo2_waveform):
    labels = []
    n_alarms = len(alarm_start_index)
    current_alarm = 0
    for i in range(len(spo2_waveform)):
        if current_alarm >= n_alarms:
            labels.append(0)
            continue
        if i < alarm_start_index[current_alarm]:  # not an alarm
            labels.append(0)
        elif alarm_start_index[current_alarm] <= i <= alarm_end_index[current_alarm]:
            if spo2_waveform[i] <= 60 or spo2_waveform[i] > 70:
                labels.append(0)
                continue
            interval_start = max(0, i - 44)
            interval_end = min(len(spo2_waveform), i+1)
            flag = 0
            for j in range(45):
                if all(60 < x <= 70 for x in spo2_waveform[interval_start: interval_end]):
                    labels.append(-1)
                    flag = 1
                    break
                else:
                    interval_start += 1
                    interval_end = min(len(spo2_waveform), interval_end + 1)
            if flag == 0:
                labels.append(0)
        elif i == alarm_end_index[current_alarm] + 1:  # not an alarm
            labels.append(0)
            current_alarm += 1
        else:  # never enter this case if data format is correct
            labels.append(0)
    return labels


# 50 < SpO2 <= 60 for <= 20 sec
def spo2_50_60(alarm_start_index, alarm_end_index, spo2_waveform):
    labels = []
    n_alarms = len(alarm_start_index)
    current_alarm = 0
    for i in range(len(spo2_waveform)):
        if current_alarm >= n_alarms:
            labels.append(0)
            continue
        if i < alarm_start_index[current_alarm]:  # not an alarm
            labels.append(0)
        elif alarm_start_index[current_alarm] <= i <= alarm_end_index[current_alarm]:
            if spo2_waveform[i] <= 50 or spo2_waveform[i] > 60:
                labels.append(0)
                continue
            interval_start = max(0, i - 19)
            interval_end = min(len(spo2_waveform), i + 1)
            flag = 0
            for j in range(20):
                if all(50 < x <= 60 for x in spo2_waveform[interval_start: interval_end]):
                    labels.append(-1)
                    flag = 1
                    break
                else:
                    interval_start += 1
                    interval_end = min(len(spo2_waveform), interval_end + 1)
            if flag == 0:
                labels.append(0)
        elif i == alarm_end_index[current_alarm] + 1:  # not an alarm
            labels.append(0)
            current_alarm += 1
        else:  # never enter this case if data format is correct
            labels.append(0)
    return labels


# |SpO2(t) - SpO2(t-10)| > 20%
def spo2_20_percent_increase(alarm_start_index, alarm_end_index, spo2_waveform):
    labels = []
    n_alarms = len(alarm_start_index)
    current_alarm = 0
    for i in range(len(spo2_waveform)):
        if current_alarm >= n_alarms:
            labels.append(0)
            continue
        if i < alarm_start_index[current_alarm]:  # not an alarm
            labels.append(0)
        elif alarm_start_index[current_alarm] <= i <= alarm_end_index[current_alarm]:
            if i < 10:
                if abs(spo2_waveform[0] - spo2_waveform[i]) > 0.2 * spo2_waveform[0]:
                    labels.append(-1)
                else:
                    labels.append(0)
            else:
                if abs(spo2_waveform[i-10] - spo2_waveform[i]) > 0.2 * spo2_waveform[i-10]:
                    labels.append(-1)
                else:
                    labels.append(0)
        elif i == alarm_end_index[current_alarm]+1:  # not an alarm
            labels.append(0)
            current_alarm += 1
        else:  # never enter this case if data format is correct
            labels.append(0)
    return labels


# |SpO2(t) - SpO2(t-5)| > 30%
def spo2_30_percent_increase(alarm_start_index, alarm_end_index, spo2_waveform):
    labels = []
    n_alarms = len(alarm_start_index)
    current_alarm = 0
    for i in range(len(spo2_waveform)):
        if current_alarm >= n_alarms:
            labels.append(0)
            continue
        if i < alarm_start_index[current_alarm]:  # not an alarm
            labels.append(0)
        elif alarm_start_index[current_alarm] <= i <= alarm_end_index[current_alarm]:
            if i < 5:
                if abs(spo2_waveform[0] - spo2_waveform[i]) > 0.3 * spo2_waveform[0]:
                    labels.append(-1)
                else:
                    labels.append(0)
            else:
                if abs(spo2_waveform[i - 5] - spo2_waveform[i]) > 0.3 * spo2_waveform[i - 5]:
                    labels.append(-1)
                else:
                    labels.append(0)
        elif i == alarm_end_index[current_alarm] + 1:  # not an alarm
            labels.append(0)
            current_alarm += 1
        else:  # never enter this case if data format is correct
            labels.append(0)
    return labels

