import pandas
from tensorflow import keras
import pandas as pd
import methcomp
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score
from datetime import datetime as date
import csv
import joblib
from preprocessing.utils import prepare_dataset, draw_error_grids, get_high_low_predictions, \
    split_data_by_hyper_into_glucose, draw_error_grids, get_error_zones_profiles

model_type = 2  # 2 2 2
epochs = 15  # 10 15 20
batch_size = 32  # 32 32 64
min_max = False

scaler = joblib.load('./models/kmeans_minimax_{}_scaler.gz'.format(min_max))
kmeans = joblib.load('./models/kmeans_minimax_{}.joblib'.format(min_max))


def get_prediction_from_DNN(X, Y, high=False):
    batch_size_kmeans = 32
    epochs_kmeans = 2000
    model_type_kmeans = 3
    min_max_kmeans = False

    if high:
        kmeans_regressor_path = './models/kmeans_regressor_index_{}_model_type_{}_epoch_{}_batch_{}_minimax_{}'
    else:
        kmeans_regressor_path = './models/kmeans_regressor_low_index_{}_model_type_{}_epoch_{}_batch_{}_minimax_{}'
    return get_high_low_predictions(X, Y, kmeans,
                                    kmeans_regressor_path, model_type_kmeans, epochs_kmeans, batch_size_kmeans,
                                    min_max_kmeans)


def load_and_eval_model(model_path, csv_file_path):
    model = keras.models.load_model(model_path)

    # model.summary()
    data = prepare_dataset(csv_file_path)
    data.drop_duplicates(subset=['sex', 'age', 'height', 'weight',
                                 'heartrate', 'bpm', 'ibi',
                                 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2',
                                 's', 'sd1/sd2', 'breathingrate', 'bloodGlucose'], keep='first', inplace=True)
    comp_data_X_og = data.loc[:, 'sex':'bloodGlucose']
    # comp_data_Y['normal'] = pd.DataFrame(np.where(comp_data_X['bloodGlucose'] < 10, 1, 0))
    temp_data_Y_og = pandas.DataFrame(columns=['bloodGlucose', 'index'])
    temp_data_Y_og['bloodGlucose'] = comp_data_X_og['bloodGlucose']
    comp_data_X_og.drop('bloodGlucose', axis=1, inplace=True)

    # print('df_X.columns', comp_data_X.columns)
    # comp_data_X = (comp_data_X - comp_data_X.mean() / comp_data_X.std()) if (comp_data_X - comp_data_X.mean() /
    # comp_data_X.std()) else 43.662943

    comp_data_X = scaler.transform(comp_data_X_og)
    comp_data_X = pandas.DataFrame(comp_data_X)
    temp_data_Y_og['index'] = range(1, len(temp_data_Y_og.index) + 1)
    comp_data_X_og['og_gluc'] = temp_data_Y_og['bloodGlucose']

    # predictions_binary = comp_data_Y_og['hyper'].values
    predictions_binary = model.predict(comp_data_X)
    predictions_binary = [1 if pred[0] >= 0.5 else 0 for pred in predictions_binary]

    data_frames_X, data_frames_Y, dfs_index = split_data_by_hyper_into_glucose(comp_data_X.values,
                                                                               temp_data_Y_og.values,
                                                                               predictions_binary)

    index = 0
    final_df = pandas.DataFrame(columns=['final_predictions', 'glucose', 'index'])

    for data_frame_X, data_frame_Y, df_ind in zip(data_frames_X, data_frames_Y, dfs_index):
        data_frame_Y = pd.DataFrame(data_frame_Y, columns=['glucose'])
        data_frame_Y['index'] = [int(ind[0]) for ind in df_ind]

        final_predictions_temp_, final_glucose_Y, df_ind = \
            get_prediction_from_DNN(data_frame_X, data_frame_Y.values, high=(index == 1))
        final_predictions_temp = pandas.DataFrame(final_glucose_Y, columns=['glucose'])
        final_predictions_temp['final_predictions'] = final_predictions_temp_
        final_predictions_temp['index'] = df_ind

        # data_frame_Y_temp = pandas.DataFrame(data_frame_Y, columns=['glucose'])

        final_df = final_df.append(final_predictions_temp)

        # final_df.reset_index(inplace=True)
        # final_df['glucose'] = final_df.append(final_predictions_temp['glucose'])

        index += 1

    draw_error_grids(final_df['glucose'], final_df['final_predictions'],
                     '-model_mega_nets')

    count_clarke_zones, count_parkes_zones = get_error_zones_profiles(final_df['glucose'],
                                                                      final_df['final_predictions'])
    comp_data_X_og['index'] = temp_data_Y_og['index']

    out_ready_df = pd.merge(comp_data_X_og, final_df, on='index', how='outer')

    out_ready_df.drop('index', axis=1, inplace=True)
    out_ready_df.drop('glucose', axis=1, inplace=True)
    out_ready_df.rename(columns={'og_gluc': 'Glucometer'}, inplace=True)
    out_ready_df.rename(columns={'final_predictions': 'Predictions'}, inplace=True)
    out_ready_df.to_csv('datasets/mega_nets_results.csv', index=False)
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


def evaluate_d_zone(model_path, csv_file_path):
    model = keras.models.load_model(model_path)

    # model.summary()
    data_p = prepare_dataset(csv_file_path, hba1c=True)

    data_p.drop_duplicates(subset=['sex', 'age', 'height', 'weight',
                                   'heartrate', 'bpm', 'ibi',
                                   'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2',
                                   's', 'sd1/sd2', 'breathingrate', 'bloodGlucose'], keep='first', inplace=True)

    # print(data_p['age'].unique())
    hba1c_df = pd.DataFrame(columns=['index', 'hba1c'])
    hba1c_df['hba1c'] = data_p['hba1c']
    data_p.drop('hba1c', axis=1, inplace=True)

    final_analysis_df = []
    for df_index in range(0, len(data_p)):
        data = data_p.iloc[[df_index]]
        comp_data_X_og = data.loc[:, 'sex':'bloodGlucose']
        # comp_data_Y['normal'] = pd.DataFrame(np.where(comp_data_X['bloodGlucose'] < 10, 1, 0))
        temp_data_Y_og = pandas.DataFrame(columns=['bloodGlucose', 'index'])
        temp_data_Y_og['bloodGlucose'] = comp_data_X_og['bloodGlucose']
        comp_data_X_og.drop('bloodGlucose', axis=1, inplace=True)

        comp_data_X = scaler.transform(comp_data_X_og)
        comp_data_X = pandas.DataFrame(comp_data_X)
        temp_data_Y_og['index'] = [df_index + 1]
        comp_data_X_og['og_gluc'] = temp_data_Y_og['bloodGlucose']

        # predictions_binary = comp_data_Y_og['hyper'].values
        predictions_binary = model.predict(comp_data_X)
        predictions_binary = [1 if pred[0] >= 0.5 else 0 for pred in predictions_binary]

        data_frames_X, data_frames_Y, dfs_index = split_data_by_hyper_into_glucose(comp_data_X.values,
                                                                                   temp_data_Y_og.values,
                                                                                   predictions_binary)

        # data_frames_X = [x for x in data_frames_X if x != []]
        # data_frames_Y = [x for x in data_frames_Y if x != []]
        # dfs_index = [x for x in dfs_index if x != []]

        index = 0
        final_df = pandas.DataFrame(columns=['final_predictions', 'glucose', 'index'])

        for data_frame_X, data_frame_Y, df_ind in zip(data_frames_X, data_frames_Y, dfs_index):
            if len(data_frame_X) == 0:
                index += 1
                continue
            data_frame_Y = pd.DataFrame(data_frame_Y, columns=['glucose'])
            data_frame_Y['index'] = [int(ind[0]) for ind in df_ind]
            final_predictions_temp_, final_glucose_Y, df_ind = \
                get_prediction_from_DNN(data_frame_X, data_frame_Y.values, high=(index == 1))
            final_predictions_temp = pandas.DataFrame(final_glucose_Y, columns=['glucose'])
            final_predictions_temp['final_predictions'] = final_predictions_temp_
            final_predictions_temp['index'] = df_ind

            final_df = final_df.append(final_predictions_temp)

            index += 1

        count_clarke_zones, count_parkes_zones = get_error_zones_profiles(final_df['glucose'],
                                                                          final_df['final_predictions'])

        if 'D' in dict(count_clarke_zones) and dict(count_clarke_zones)['D'] != 0:
            final_analysis_df.append([1 if final_df.loc[0]['glucose'] else 0, predictions_binary[0],
                                      hba1c_df.loc[df_index]['hba1c'], final_df.loc[0]['glucose']])

    print(len(final_analysis_df))
    columns = ['actual', 'hyper', 'hba1c', 'glucometer glucose']
    df = pd.DataFrame(final_analysis_df, columns=columns)
    df.to_csv('datasets/final_analysis_df.csv')


if __name__ == '__main__':
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

    load_and_eval_model('./models/binary_type_{}_epoch_{}_batch_{}'.format(model_type, epochs, batch_size),
                        'datasets/combined_tests.csv')
