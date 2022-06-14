from tensorflow import keras
import pandas as pd
import numpy as np
import json
from datetime import datetime as date
import csv
import joblib


def calculate_age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def prepare_single_row(dataset_path):
    new_data = []
    count = 0
    with open(dataset_path) as csv_file:
        csv_reader = csv.reader(csv_file)

        for row in csv_reader:
            newRow = []
            # ignore headers
            if count < 1:
                count += 1
                continue

            gg = row[3]

            # remove this check to get all records
            if count > 1:
                break

            count += 1
            # replacing quotes
            row[1] = row[1].replace("'", '"').replace('True', '"True"').replace('False', '"False"')

            # Profile
            if row[1]:
                temp = json.loads(row[1])
                if 'sex' in temp:
                    newRow.append(temp['sex'])
                if 'birthday' in temp:
                    age = calculate_age(date.fromtimestamp(abs(temp['birthday']) / 1000))
                    newRow.append(age)
                if 'height(cm)' in temp:
                    newRow.append(float(temp['height(cm)']))
                if 'weight(kg)' in temp:
                    newRow.append(float(temp['weight(kg)']))
            newRow.append(row[2])  # heartrate
            for index in range(14, len(row)):
                newRow.append(float(row[index]) if row[index] and row[index] != '--' and
                                                   row[index] != 'nan' and row[index] != 'inf' else 0)
            if newRow[len(newRow) - 1] != float(0):
                new_data.append(newRow)

    df = pd.DataFrame(new_data, columns=['sex', 'age', 'height', 'weight',
                                         'heartrate', 'bpm', 'ibi',
                                         'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2',
                                         's', 'sd1/sd2', 'breathingrate'])

    # converting categorical features into numerical
    categorical_features = [('sex', 'Male')]
    for cf in categorical_features:
        df[cf[0]] = np.where(df[cf[0]] == cf[1], 1, 0)

    # saving to csv
    # df.to_csv(dataset_path[:len(dataset_path) - 4] + '_cleaned.csv', index=False)

    # print(df.head())
    for col_name in df.columns:
        df[col_name] = df[col_name].astype(float)
    return df, gg


# HPs used for model naming
model_type = 2
epochs = 15
batch_size = 32

# paths to data files/models
csv_file_path = './datasets/2173747691999486503022439312933411749allrecord_for_update_hyper_mode.csv'
model_path = './models/binary_type_{}_epoch_{}_batch_{}'.format(model_type, epochs, batch_size)

# load model
model = keras.models.load_model(model_path)
print(model.summary())

# returning 1st row
single_row_df, glucometer_glucose = prepare_single_row(csv_file_path)

# display for sanity check
pd.options.display.max_columns = 200
pd.options.display.max_rows = 200
print(single_row_df)

# normalization
scaler = joblib.load(model_path + '_scaler.gz')
single_row_df = scaler.transform(single_row_df)

# run prediction on dataframe
predictions = model.predict(single_row_df)

# add a threshold to convert to classes
predict_single = ['hyper' if pred[0] >= 0.5 else 'normal' for pred in predictions]

print('\n\nPredictions:')
print('Glucometer Value', glucometer_glucose)
print('predicted score', predictions[0][0])
print('predicted class', predict_single[0])
