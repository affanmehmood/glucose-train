import csv
import pandas as pd
import json
import numpy as np
from datetime import datetime as date
new_data = []

count = 0
skipCount = 0


def get_real_value(arg1, arg2):
    if arg1 is not None:
        return arg1
    else:
        return arg2


def get_real_target(arg1, arg2):
    if arg1 is not None:
        return arg1 if arg1 else -1
    else:
        return arg2 if arg2 else -1


def calculate_age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


with open('../datasets/allrecord.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        newRow = []
        profileDict = {
            'user_id': None,
            'age': None,
            'sex': None,
            'height': None,
            'weight': None,
            # 'diabetic': None,
            # 'smoker': None,
            # 'hbA1C': None,
            # 'diabeticType': None,
            'heartrate': None
        }
        # ignore headers
        if count == 0:
            count += 1
            continue
        count += 1

        # replacing quotes
        for i in range(1, len(row)):
            row[i] = row[i].replace("'", '"').replace('True', '"True"').replace('False', '"False"')

        profileDict['userkey'] = row[1]
        # Profile
        if row[3]:
            temp = json.loads(row[3])
            if 'birthday' in temp:
                profileDict['age'] = temp['birthday']
            if 'sex' in temp:
                profileDict['sex'] = temp['sex']
            if 'height(cm)' in temp:
                profileDict['height'] = temp['height(cm)']
            if 'weight(kg)' in temp:
                profileDict['weight'] = temp['weight(kg)']
            # if 'diabetes' in temp:
            #     profileDict['diabetic'] = temp['diabetes']
            # else:
            #     profileDict['diabetic'] = 'False'
            # if 'smoke' in temp:
            #     profileDict['smoker'] = temp['smoke']
            # if 'hbA1C' in temp:
            #     profileDict['hbA1C'] = temp['hbalc']

        # appending features
        newRow.append(profileDict['userkey'])
        newRow.append(profileDict['sex'])
        newRow.append(calculate_age(date.fromtimestamp(abs(profileDict['age'])/1000)))
        newRow.append(profileDict['height'])
        newRow.append(profileDict['weight'])
        # newRow.append(profileDict['diabetic'])
        # newRow.append(profileDict['smoker'])
        # newRow.append(profileDict['hbA1C'])
        # try:
        #     newRow.append('type2' if float(row[6]) > 9 else 'normal')
        # except ValueError:
        #     newRow.append(row[6])

        newRow.append(row[4])

        for i in range(7, 20):
            newRow.append(float(row[i]) if row[i] and row[i] != '--' and
                                           row[i] != 'nan' and row[i] != 'inf' else 0)  # [last single records]
        # target
        try:
            newRow.append(float(row[6]))
        except ValueError:
            newRow.append(0)
        new_data.append(newRow)

df = pd.DataFrame(new_data, columns=['userkey', 'sex', 'age', 'height', 'weight',
                                     'heartrate', 'bpm', 'ibi',
                                     'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2',
                                     's', 'sd1/sd2', 'breathingrate', 'bloodGlucose'])

# dropping useless rows
# df = df.dropna(subset=['bloodGlucose'])  # drop nan target values
# df = df[df['bloodGlucose'] > 0]

# Putting mean values inplace of -1, None and 0
possible_missing_floats = ['height', 'weight',  # 'hbA1C',
                           'heartrate', 'bpm', 'ibi',
                           'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2',
                           's', 'sd1/sd2', 'breathingrate']
for key in possible_missing_floats:
    mean = df[key].mean()
    df.loc[df[key] == -1, key] = mean
    df.loc[df[key] == 0, key] = mean
    df[key].fillna(value=mean, inplace=True)

# converting categorical features into numerical
categorical_features = [('sex', 'Male')]
for cf in categorical_features:
    df[cf[0]] = np.where(df[cf[0]] == cf[1], 1, 0)

og_len = len(df)
# drop duplicates
df.drop_duplicates(keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
print('dropped rows', og_len - len(df))
# saving to csv
df.to_csv('../datasets/allrecord_cleaned.csv', index=False)

# todo visualize and inspect
