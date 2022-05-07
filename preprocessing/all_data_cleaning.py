import csv
import pandas as pd
import json
import numpy as np

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


with open('../datasets/enne.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        newRow = []
        profileDict = {
            'user_id': None,
            'sex': None,
            'age': None,
            'height': None,
            'weight': None,
            'waist': None,
            'diabetic': None,
            'smoker': None,
            'hbA1C': None,
            'diabeticType': None
        }
        eventsDict = {
            'bloodGlucose': None,
            'systolic': None,
            'diastolic': None,
            'heartrate': None,
            'spo2': None,
            'temperature': None
        }
        dataFeaturesDict = {
            'heartrate': None,
            'systolic': None,
            'diastolic': None,
            'bloodGlucose': None,
        }
        # ignore headers
        if count == 0:
            count += 1
            continue
        count += 1

        # replacing quotes
        for i in range(1, len(row)):
            row[i] = row[i].replace("'", '"').replace('True', '"True"').replace('False', '"False"')

        profileDict['userkey'] = row[2]
        # Profile
        if row[3]:
            temp = json.loads(row[3])
            if 'sex' in temp:
                profileDict['sex'] = temp['sex']
            if 'age' in temp:
                profileDict['age'] = temp['age']
            if 'height' in temp:
                profileDict['height'] = temp['height']
            if 'weight' in temp:
                profileDict['weight'] = temp['weight']
            if 'waist' in temp:
                profileDict['waist'] = temp['waist']
            if 'diabetic' in temp:
                profileDict['diabetic'] = temp['diabetic']
            if 'smoker' in temp:
                profileDict['smoker'] = temp['smoker']
            if 'hbA1C' in temp:
                profileDict['hbA1C'] = temp['hbA1C']
            if 'diabeticType' in temp:
                profileDict['diabeticType'] = temp['diabeticType']
        # Events
        if row[4]:
            temp = json.loads(row[4])
            if temp['glucometer']:
                eventsDict['bloodGlucose'] = temp['glucometer']['bloodGlucose']
            if 'bloodPressure' in temp:
                eventsDict['systolic'] = temp['bloodPressure']['systolic']
                eventsDict['diastolic'] = temp['bloodPressure']['diastolic']
                eventsDict['temperature'] = temp['temperature']
            if 'heartrate' in temp:
                eventsDict['heartrate'] = temp['heartrate']
            if 'spo2' in temp:
                eventsDict['spo2'] = temp['spo2']
        # Data->Features
        if row[8]:
            temp = json.loads(row[8])
            if 'features' in temp and len(temp['features'][0]) > 0:
                dataFeaturesDict['bloodGlucose'] = temp['features'][0]['bloodGlucose']
                dataFeaturesDict['heartrate'] = temp['features'][0]['heartrate']
                dataFeaturesDict['systolic'] = temp['features'][0]['bloodPressure']['systolic']
                dataFeaturesDict['diastolic'] = temp['features'][0]['bloodPressure']['diastolic']

        # appending features
        newRow.append(profileDict['userkey'])
        newRow.append(profileDict['sex'])
        newRow.append(profileDict['age'])
        newRow.append(profileDict['height'])
        newRow.append(profileDict['weight'])
        newRow.append(profileDict['waist'])
        newRow.append(profileDict['diabetic'])
        newRow.append(profileDict['smoker'])
        newRow.append(profileDict['hbA1C'])
        newRow.append('type2' if get_real_target(dataFeaturesDict['bloodGlucose'],
                                                 eventsDict['bloodGlucose']) > 9 else 'normal')

        newRow.append(get_real_value(dataFeaturesDict['diastolic'], eventsDict['diastolic']))
        newRow.append(get_real_value(dataFeaturesDict['systolic'], eventsDict['systolic']))
        newRow.append(get_real_value(dataFeaturesDict['heartrate'], eventsDict['heartrate']))

        for i in range(9, 22):
            newRow.append(float(row[i]) if row[i] and row[i] != '--' and
                                           row[i] != 'nan' and row[i] != 'inf' else 0)  # [last single records]
        # target
        newRow.append(get_real_value(dataFeaturesDict['bloodGlucose'], eventsDict['bloodGlucose']))
        new_data.append(newRow)

df = pd.DataFrame(new_data, columns=['userkey', 'sex', 'age', 'height', 'weight', 'waist', 'diabetic', 'smoker', 'hbA1C',
                                     'diabeticType', 'diastolic', 'systolic', 'heartrate', 'bpm', 'ibi',
                                     'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2',
                                     's', 'sd1/sd2', 'breathingrate', 'bloodGlucose'])

# dropping useless rows
df = df.dropna(subset=['bloodGlucose'])  # drop nan target values
df = df[df['bloodGlucose'] > 0]

# Putting mean values inplace of -1, None and 0
possible_missing_floats = ['age', 'height', 'weight', 'waist', 'hbA1C',
                           'diastolic', 'systolic', 'heartrate', 'bpm', 'ibi',
                           'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2',
                           's', 'sd1/sd2', 'breathingrate']
for key in possible_missing_floats:
    mean = df[key].mean()
    df.loc[df[key] == -1, key] = mean
    df.loc[df[key] == 0, key] = mean
    df[key].fillna(value=mean, inplace=True)

# converting categorical features into numerical
categorical_features = [('sex', 'male'), ('diabetic', 'True'), ('smoker', 'True'), ('diabeticType', 'type2')]
for cf in categorical_features:
    df[cf[0]] = np.where(df[cf[0]] == cf[1], 1, 0)

og_len = len(df)
# drop duplicates
df.drop_duplicates(keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)
print('dropped models', og_len - len(df))
# drop unwanted columns
df.drop("diabetic", axis=1, inplace=True)
df.drop("smoker", axis=1, inplace=True)
df.drop("diabeticType", axis=1, inplace=True)
df.drop("diastolic", axis=1, inplace=True)
df.drop("hbA1C", axis=1, inplace=True)
df.drop("waist", axis=1, inplace=True)
df.drop("systolic", axis=1, inplace=True)

print(df.columns)
# saving to csv
df.to_csv("../datasets/cleaned_enne.csv", index=False)

# todo visualize and inspect
