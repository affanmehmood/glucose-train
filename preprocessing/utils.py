from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime as date
import csv
import json
import numpy as np
import methcomp
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from tensorflow import keras


def scale_data_minmax(trainX, testX):
    input_scaler = MinMaxScaler()
    input_scaler.fit(trainX)
    # transform training dataset
    trainX = input_scaler.transform(trainX)
    # transform test dataset
    testX = input_scaler.transform(testX)
    return trainX, testX, input_scaler


def scale_data_std(trainX, testX=None):
    # print(testX.columns)
    # print(trainX.columns)
    input_scaler = StandardScaler()
    input_scaler.fit(trainX)
    # transform training dataset
    trainX = input_scaler.transform(trainX)
    # transform test dataset
    if testX is None:
        return trainX, input_scaler
    else:
        testX = input_scaler.transform(testX)
        return trainX, testX, input_scaler


def calculate_age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


'''
    This function for preprocessing on the fly
    Code that create a new cleaned csv are
    in preprocessing with the word 'cleaning' in them
'''


def prepare_dataset(dataset_path, min_max=False, hba1c=False):
    new_data = []
    count = 0
    with open(dataset_path) as csv_file:
        csv_reader = csv.reader(csv_file)

        for row in csv_reader:
            newRow = []
            # ignore headers
            if count == 0:
                count += 1
                continue
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
            if min_max:
                mini, maxi = get_min_max(row[13])
                newRow.append(mini)
                newRow.append(maxi)

            for index in range(14, len(row)):
                newRow.append(float(row[index]) if row[index] and row[index] != '--' and
                                                   row[index] != 'nan' and row[index] != 'inf' else 0)

            if hba1c:
                if row[1]:
                    temp = json.loads(row[1])
                    if 'hbalc' in temp:
                        newRow.append(float(temp['hbalc']))
                    else:
                        newRow.append(float(0))

            newRow.append(float(row[3]) if row[3] else 0)
            if newRow[len(newRow) - 1] != float(0):
                new_data.append(newRow)

    if min_max:
        columns = ['sex', 'age', 'height', 'weight',
                   'heartrate', 'min', 'max', 'bpm', 'ibi',
                   'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2',
                   's', 'sd1/sd2', 'breathingrate', 'bloodGlucose']
    else:
        columns = ['sex', 'age', 'height', 'weight',
                   'heartrate', 'bpm', 'ibi',
                   'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2',
                   's', 'sd1/sd2', 'breathingrate', 'bloodGlucose']
    if hba1c:
        columns.insert(len(columns) - 1, 'hba1c')

    df = pd.DataFrame(new_data, columns=columns)

    # converting categorical features into numerical
    categorical_features = [('sex', 'Male')]
    for cf in categorical_features:
        df[cf[0]] = np.where(df[cf[0]] == cf[1], 1, 0)

    for col_name in df.columns:
        df[col_name] = df[col_name].astype(float)
    return df


def get_min_max(min_max):
    mini, maxi = min_max.split('/')
    return float(mini), float(maxi)


def prepare_data_splits_from_dataframe(df):
    # 70% of records for generating the training set
    train_len = int(len(df) * 0.7)

    # Remaining 30% of records for generating the evaluation and serving sets
    eval_test_len = len(df) - train_len

    # Half of the 30%, which makes up 15% of total records, for generating the evaluation set
    eval_len = eval_test_len // 2

    # Remaining 15% of total records for generating the test set
    test_len = eval_test_len - eval_len

    # Sample the train, validation and serving sets. We specify a random state for repeatable outcomes.
    train_df = df.iloc[:train_len].sample(frac=1, random_state=48).reset_index(drop=True)
    eval_df = df.iloc[train_len: train_len + eval_len].sample(frac=1, random_state=48).reset_index(drop=True)
    test_df = df.iloc[train_len + eval_len: train_len + eval_len + test_len].sample(
        frac=1, random_state=48).reset_index(drop=True)

    # Testing data emulates the data that would be submitted for predictions, so it should not have the label column.
    # test_df = test_df.drop(['bloodGlucose'], axis=1)

    return train_df, eval_df, test_df


def draw_error_grids(Y_test_users, y_pred_users, title):
    plt.figure(figsize=(12, 12))
    plot = methcomp.glucose.parkes(type=2, reference=np.array(Y_test_users.values * 18), units='mg/dl',
                                   test=np.array(y_pred_users * 18), color_points='black')
    plot.figure.savefig('./saved_figs/Parkes error plot - ' + title + '.png')

    plt.figure(figsize=(12, 12))
    plot = methcomp.glucose.clarke(reference=np.array(Y_test_users.values * 18), units='mg/dl',
                                   test=np.array(y_pred_users * 18), color_points='black')
    plot.figure.savefig('./saved_figs/Clarkes error plot - ' + title + '.png')


def get_error_zones_profiles(Y_test_users, y_pred_users):
    parkes_zones = methcomp.glucose.parkeszones(type=2, reference=np.array(Y_test_users), units='mmol',
                                                test=np.array(y_pred_users), numeric=False)
    parkes_zones = Counter(parkes_zones)

    clarke_zones = methcomp.glucose.clarkezones(reference=np.array(Y_test_users), units='mmol',
                                                test=np.array(y_pred_users), numeric=False)
    clarke_zones = Counter(clarke_zones)

    return clarke_zones, parkes_zones


# function that splits the data into clusters
def split_data_by_hyper_into_glucose(data_frameX, data_frameY, result_mask):
    dfsX = [[] for _ in np.bincount(result_mask)]
    dfsY = [[] for _ in np.bincount(result_mask)]

    # to keep track of index
    dfs_index = [[] for _ in np.bincount(result_mask)]

    index_array = data_frameY[:, -1:]
    data_frameY = data_frameY[:, :-1]
    # print(data_frameX[len(data_frameX) - 1])

    for index, mask in enumerate(result_mask):
        dfsX[mask].append(data_frameX[index])
        dfsY[mask].append(data_frameY[index])
        dfs_index[mask].append(index_array[index])
    return dfsX, dfsY, dfs_index


# function that splits the data into clusters
def split_data_by_kmeans(data_frameX, data_frameY, kmeans):
    result_mask = kmeans.predict(data_frameX)

    dfsX = [[] for _ in np.bincount(result_mask)]
    dfsY = [[] for _ in np.bincount(result_mask)]

    # to keep track of index
    dfs_index = [[] for _ in np.bincount(result_mask)]

    # must send index as 2nd columns of Y
    index_array = data_frameY[:, -1:]
    data_frameY = data_frameY[:, :-1]

    for index, mask in enumerate(result_mask):
        dfsX[mask].append(data_frameX[index])
        dfsY[mask].append(data_frameY[index])
        dfs_index[mask].append(index_array[index])

    return dfsX, dfsY, dfs_index


# function that splits data into n clusters and applies DNNs
def get_high_low_predictions(df_complete_X, df_complete_Y, kmeans, kmeans_regressor_path, model_type,
                             epochs, batch_size, min_max):
    data_frames_X, data_frames_Y, dfs_indexs = split_data_by_kmeans(df_complete_X, df_complete_Y, kmeans)
    cluster_index = 0
    final_DF_X = []
    final_DF_Y = []
    final_DF_IDX = []

    # data_frames_X = [x for x in data_frames_X if x != []]
    # data_frames_Y = [x for x in data_frames_Y if x != []]
    # dfs_indexs = [x for x in dfs_indexs if x != []]

    for data_frame_X, data_frame_Y, dfs_index in zip(data_frames_X, data_frames_Y, dfs_indexs):
        if len(data_frame_X) == 0:
            cluster_index += 1
            continue
        model = keras.models.load_model(kmeans_regressor_path.format(cluster_index, model_type,
                                                                     epochs, batch_size, min_max))

        data_frame_X = pd.DataFrame(data_frame_X)
        data_frame_Y = pd.DataFrame(data_frame_Y)
        data_frame_idx = pd.DataFrame(dfs_index)

        predictions = model.predict(data_frame_X)

        for index, prediction in enumerate(predictions):
            final_DF_X.append(prediction)
            final_DF_Y.append(data_frame_Y.iloc[index])
            final_DF_IDX.append(data_frame_idx.iloc[index])
        cluster_index += 1
    predictions = [pred[0] for pred in final_DF_X]
    glucose_Y = [gluc[0] for gluc in final_DF_Y]
    dfs_index = [index[0] for index in final_DF_IDX]

    return predictions, glucose_Y, dfs_index
