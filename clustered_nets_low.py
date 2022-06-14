import os.path

import pandas
from tensorflow import keras
import pandas as pd
import tensorflow as tf
import methcomp
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.utils import scale_data_std
from datetime import datetime as date
import joblib
import csv
import json
from preprocessing.utils import prepare_dataset, draw_error_grids, get_error_zones_profiles, \
    prepare_data_splits_from_dataframe, split_data_by_kmeans, get_high_low_predictions

batch_size = 32
epochs = 2000
model_type = 3
min_max = False

### Best so far (non minmax) ###
# batch_size = 32
# epochs = 2000
# model_type = 3
# min_max = False
scaler = joblib.load('./models/kmeans_minimax_{}_scaler.gz'.format(min_max))
kmeans = joblib.load('./models/kmeans_minimax_{}.joblib'.format(min_max))
kmeans_regressor_path = './models/kmeans_regressor_low_index_{}_model_type_{}_epoch_{}_batch_{}_minimax_{}'

# Select labels for inputs and outputs.
if min_max:
    inputs = ['sex', 'age', 'height', 'weight',
              'heartrate', 'min', 'max', 'bpm', 'ibi',
              'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2',
              's', 'sd1/sd2', 'breathingrate']
else:
    inputs = ['sex', 'age', 'height', 'weight',
              'heartrate', 'bpm', 'ibi',
              'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2',
              's', 'sd1/sd2', 'breathingrate']
outputs = ["bloodGlucose"]


def get_model_3(inputs):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(len(inputs),), name="input"),
        keras.layers.Dense(32, activation="relu", name="dense_1"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation="relu", name="dense_2"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(128, activation="relu", name="dense_3"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation="sigmoid", name="dense_4"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu", name="dense_5"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation="relu", name="dense_6"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(32, activation="relu", name="dense_7"),
        keras.layers.Dense(1, name="output"),
    ], name="model")
    return model


def main_exp():
    if min_max:
        train_df = pd.read_csv("datasets/cleaned_enne_minmax.csv")
    else:
        train_df = pd.read_csv("datasets/cleaned_enne.csv")

    train_df = train_df.loc[train_df['bloodGlucose'] < 10]
    # train_df, eval_df, test_df = prepare_data_splits_from_dataframe(data)
    # split target from train
    train_X = train_df.loc[:, 'sex':'breathingrate']
    train_Y = train_df.loc[:, 'bloodGlucose':'bloodGlucose']
    train_X.reset_index(drop=True, inplace=True)
    train_Y.reset_index(drop=True, inplace=True)

    # Normalize data
    train_X = scaler.transform(train_X)

    data_frames_X, data_frames_Y, = split_data_by_kmeans(train_X, train_Y.values, kmeans)

    index = 0
    for data_frame_X, data_frame_Y in zip(data_frames_X, data_frames_Y):

        data_frame_X = pandas.DataFrame(data_frame_X)
        data_frame_Y = pandas.DataFrame(data_frame_Y)

        if model_type == 3:
            model = get_model_3(inputs)

        model.compile(optimizer="adam", loss='mean_absolute_error',
                      metrics=['mean_absolute_error'])

        model.fit(data_frame_X, data_frame_Y, batch_size=batch_size, epochs=epochs,
                  verbose=True)
        # save the model
        model.save('./models/kmeans_regressor_low_index_{}_model_type_{}_epoch_{}_batch_{}_minimax_{}'
                   .format(index, model_type, epochs, batch_size, min_max))
        index += 1


def load_and_eval_model_trail_new(model_path, csv_file_path, csv_num, clusters=3):
    data = prepare_dataset(csv_file_path, min_max=min_max)
    data.drop_duplicates(subset=['sex', 'age', 'height', 'weight',
                                 'heartrate', 'bpm', 'ibi',
                                 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2',
                                 's', 'sd1/sd2', 'breathingrate', 'bloodGlucose'], keep='first', inplace=True)
    data = data.loc[data['bloodGlucose'] < 10]
    comp_data_X = data.loc[:, 'sex':'bloodGlucose']
    comp_data_Y = pd.DataFrame()
    comp_data_Y['bloodGlucose'] = comp_data_X['bloodGlucose']
    comp_data_X.drop('bloodGlucose', axis=1, inplace=True)

    # normalize
    print(model_path)
    comp_data_X = scaler.transform(comp_data_X)

    comp_data_Y['index'] = range(1, len(comp_data_Y.index) + 1)
    predictions, glucose_Y, df_ind = get_high_low_predictions(comp_data_X, comp_data_Y.values, kmeans,
                                                              kmeans_regressor_path, model_type, epochs, batch_size,
                                                              min_max)

    og_df = pd.DataFrame()
    # og_df.drop_duplicates(subset=['sex', 'age', 'height', 'weight',
    #                              'heartrate', 'bpm', 'ibi',
    #                              'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2',
    #                              's', 'sd1/sd2', 'breathingrate', 'bloodGlucose'], keep='first', inplace=True)
    # og_df = og_df.loc[og_df['Glucometer Values'] < 10]
    # og_df = og_df[og_df['Glucometer Values'].notna()]
    og_df['Glucometer Original'] = glucose_Y
    og_df['Predictions'] = predictions
    # og_df.to_excel(csv_file_path[:len(csv_file_path) - 4] + 'K-means-Reg-PREDS.xlsx')

    draw_error_grids(og_df['Glucometer Original'], og_df['Predictions'],
                     '-low-k_means_model')
    count_clarke_zones, count_parkes_zones = get_error_zones_profiles(og_df['Glucometer Original'],
                                                                      og_df['Predictions'])
    return count_clarke_zones, count_parkes_zones


if __name__ == '__main__':
    # main_exp()
    csv_paths = [
        'datasets/-4580253642453236142-466387376852218172alltrialrecord.csv',
        'datasets/-8845707664459066312alltrialdata-05-16_05-18.csv',
        'datasets/2173747691999486503022439312933411749allrecord_for_update_hyper_mode.csv',
        'datasets/-47054135960479185alltridata-05-19_05-20.csv',
        # 'datasets/-7791009196018876848alltridata-05-19_05-20.csv',
        'datasets/4792894170021136961alltrialdata-05-18_05-19-T2.csv'
    ]

    combined_df = pd.concat(
        map(pd.read_csv, csv_paths), ignore_index=True)
    combined_df.to_csv('datasets/combined_tests.csv', index=False)

    count_clarke_zones, count_parkes_zones = load_and_eval_model_trail_new('./models/kmeans_regressor_index_{}'
                                                               '_model_type_{}_epoch_{}_batch_{}_minimax_{}',
                                                               'datasets/combined_tests.csv', 1)
    temp_dict = {
        'A': 0,
        'B': 0,
        'C': 0,
    }
    for key in dict(count_parkes_zones).keys():
        temp_dict[key] = dict(count_parkes_zones)[key]
    print('count_parkes_zones', temp_dict)

    temp_dict = {
        'A': 0,
        'B': 0,
        'C': 0,
    }
    for key in dict(count_clarke_zones).keys():
        temp_dict[key] = dict(count_clarke_zones)[key]
    print('count_clarke_zones', temp_dict)


def get_low_predictions(X, Y):  # wrapper function for the pipeline
    return get_high_low_predictions(X, Y, kmeans,
                                    kmeans_regressor_path, model_type, epochs, batch_size, min_max)
