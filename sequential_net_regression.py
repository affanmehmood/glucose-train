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
from preprocessing.utils import prepare_dataset, draw_error_grids, prepare_data_splits_from_dataframe

# batch_size = 16
# epochs = 2000
# model_type = 2
# min_max = True

### Best so far (non minmax) ###
batch_size = 32
epochs = 2000
model_type = 3
min_max = True


def get_model_1(inputs):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(len(inputs),), name="input"),
        keras.layers.Dense(512, activation="relu", name="dense_1"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(512, activation="relu", name="dense_2"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1024, name="dense_3"),
        keras.layers.Dropout(0.1),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(1024, name="dense_4"),
        keras.layers.Dropout(0.2),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(2048, name="dense_5"),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(1024, name="dense_6"),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(1024, name="dense_7"),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(512, name="dense_8"),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(256, activation="relu", name="dense_9"),
        # keras.layers.LeakyReLU(),
        # keras.layers.Dense(64, activation="relu", name="dense_10"),
        keras.layers.Dense(1, name="output"),
    ], name="model")
    return model


def get_model_2(inputs):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(len(inputs),), name="input"),
        keras.layers.Dense(32, kernel_initializer='normal', activation="relu", name="dense_1"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, kernel_initializer='normal', activation="relu", name="dense_2"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, kernel_initializer='normal', activation="relu", name="dense_3"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, kernel_initializer='normal', activation="relu", name="dense_4"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, kernel_initializer='normal', activation="relu", name="dense_5"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(512, kernel_initializer='normal', activation="relu", name="dense_6"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(512, kernel_initializer='normal', activation="relu", name="dense_7"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_8"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_9"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_10"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(512, kernel_initializer='normal', activation="relu", name="dense_11"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, kernel_initializer='normal', activation="relu", name="dense_12"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, kernel_initializer='normal', activation="relu", name="dense_13"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, kernel_initializer='normal', activation="relu", name="dense_14"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, kernel_initializer='normal', activation="relu", name="dense_15"),
        # keras.layers.Dense(512, kernel_initializer='normal', activation="relu", name="dense_27"),
        keras.layers.Dense(1, activation='linear', name="output"),
    ], name="model")
    return model


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


def get_model_4(inputs):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(len(inputs),), name="input"),
        keras.layers.Dense(512, kernel_initializer='normal', activation="relu", name="dense_1"),
        # keras.layers.Dropout(0.1),
        keras.layers.Dense(512, kernel_initializer='normal', activation="relu", name="dense_2"),
        keras.layers.Dense(512, kernel_initializer='normal', activation="relu", name="dense_3"),
        keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_4"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_5"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_6"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_7"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_8"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_10"),
        keras.layers.Dense(4096, kernel_initializer='normal', activation="relu", name="dense_11"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(4096, kernel_initializer='normal', activation="relu", name="dense_12"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(4096, kernel_initializer='normal', activation="relu", name="dense_13"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_14"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_15"),
        keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_16"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_17"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_18"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_19"),
        keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_20"),
        keras.layers.Dense(512, kernel_initializer='normal', activation="relu", name="dense_27"),
        keras.layers.Dense(1, activation='linear', name="output"),
    ], name="model")
    return model


def main_exp():
    if min_max:
        data = pd.read_csv("datasets/cleaned_enne_minmax.csv")
    else:
        data = pd.read_csv("datasets/cleaned_enne.csv")
    # train_df, eval_df, test_df = prepare_data_splits_from_dataframe(data)
    train_df = data
    # split target from train
    train_X = data.loc[:, 'sex':'breathingrate']
    train_Y = train_df.loc[:, 'bloodGlucose':'bloodGlucose']
    train_X.reset_index(drop=True, inplace=True)
    train_Y.reset_index(drop=True, inplace=True)

    # Normalize data
    train_X, scaler = scale_data_std(train_X)
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

    if model_type == 1:
        model = get_model_1(inputs)
    elif model_type == 2:
        model = get_model_2(inputs)
    elif model_type == 3:
        model = get_model_3(inputs)
    elif model_type == 4:
        model = get_model_4(inputs)

    model.compile(optimizer="adam", loss='mean_absolute_error',
                  metrics=['mean_absolute_error'])

    # model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs,
    #           verbose=True)

    # save the model
    # model.save('./models/regressor_model_type_{}_epoch_{}_batch_{}_minimax_{}'
    #            .format(model_type, epochs, batch_size, min_max))
    joblib.dump(scaler,
                './models/regressor_model_type_{}_epoch_{}_batch_{}_minimax_{}_scaler.gz'
                .format(model_type, epochs, batch_size, min_max))


def load_and_eval_model_trail_1(model_path):
    print('evaluating trail 1')
    model = keras.models.load_model(model_path)

    csv_file_path = "datasets/allrecord.csv"
    data = prepare_dataset(csv_file_path)
    for col_name in data.columns:
        data = data[data[col_name].notna()]

    comp_data_X = data.loc[:, 'sex':'breathingrate']
    comp_data_Y = data.loc[:, 'bloodGlucose':'bloodGlucose']

    scaler = joblib.load(model_path + '_scaler.gz')
    comp_data_X = scaler.transform(comp_data_X)
    predictions = model.predict(comp_data_X)

    predictions = [pred[0] for pred in predictions]
    ground_truth = [gt[0] for gt in comp_data_Y.values]
    # print(list(zip(ground_truth, predictions)))
    mae = tf.keras.losses.MeanAbsoluteError()
    print('Mean absolute error on trail 1:', mae(ground_truth, predictions).numpy())


def load_and_eval_model_trail_2(model_path):
    print('evaluating trail 2')
    model = keras.models.load_model(model_path)

    csv_file_path = "datasets/test_cleaned.csv"
    data = prepare_dataset(csv_file_path)
    comp_data_X = data.loc[:, 'sex':'bloodGlucose']
    for col_name in comp_data_X.columns:
        comp_data_X = comp_data_X[comp_data_X[col_name].notna()]
    comp_data_Y = pd.DataFrame()
    comp_data_Y['bloodGlucose'] = comp_data_X['bloodGlucose']
    comp_data_X.drop('bloodGlucose', axis=1, inplace=True)

    # normalize
    scaler = joblib.load(model_path + '_scaler.gz')
    comp_data_X = scaler.transform(comp_data_X)

    predictions = model.predict(comp_data_X)

    predictions = [pred[0] for pred in predictions]
    ground_truth = [gt[0] for gt in comp_data_Y.values]
    # print(list(zip(ground_truth, predictions)))
    mae = tf.keras.losses.MeanAbsoluteError()
    print('Mean absolute error on trail 2:', mae(ground_truth, predictions).numpy())


def load_and_eval_model_trail_new(model_path, csv_file_path):
    model = keras.models.load_model(model_path)

    data = prepare_dataset(csv_file_path, min_max=min_max)
    comp_data_X = data.loc[:, 'sex':'bloodGlucose']
    comp_data_Y = pd.DataFrame()
    comp_data_Y['bloodGlucose'] = comp_data_X['bloodGlucose']
    comp_data_X.drop('bloodGlucose', axis=1, inplace=True)

    # normalize
    print(model_path)
    scaler = joblib.load(model_path + '_scaler.gz')
    comp_data_X = scaler.transform(comp_data_X)

    predictions = model.predict(comp_data_X)

    predictions = [pred[0] for pred in predictions]
    ground_truth = [gt[0] for gt in comp_data_Y.values]
    # print(list(zip(ground_truth, predictions)))
    mae = tf.keras.losses.MeanAbsoluteError()
    print('Mean absolute error on trail 3:', mae(ground_truth, predictions).numpy())

    og_df = pd.read_csv(csv_file_path)
    og_df = og_df[og_df['Glucometer Values'].notna()]
    og_df['Glucometer Original'] = og_df['Glucometer Values']
    og_df['Predictions'] = predictions
    og_df.to_excel(csv_file_path[:len(csv_file_path) - 4] + 'Reg-PREDS.xlsx')

    draw_error_grids(og_df['Glucometer Original'], og_df['Predictions'],
                     'model_{}_epoch_{}_batch_{}_min_max_{}'.format(model_type, epochs, batch_size, min_max))


if __name__ == '__main__':
    main_exp()
    # load_and_eval_model_trail_1('./models/regressor_model_type_{}_epoch_{}_batch_{}'
    #                             .format(model_type, epochs, batch_size))
    # load_and_eval_model_trail_2('./models/regressor_model_type_{}_epoch_{}_batch_{}'
    #                             .format(model_type, epochs, batch_size))
    csv_paths = [
        'datasets/-4580253642453236142-466387376852218172alltrialrecord.csv',
        # 'datasets/-8845707664459066312alltrialdata-05-16_05-18.csv',
        # 'datasets/2173747691999486503022439312933411749allrecord_for_update_hyper_mode.csv',
        # 'datasets/-47054135960479185alltridata-05-19_05-20.csv',
        # 'datasets/-7791009196018876848alltridata-05-19_05-20.csv',
        # 'datasets/4792894170021136961alltrialdata-05-18_05-19-T2.csv'
    ]

    for csv_path in csv_paths:
        load_and_eval_model_trail_new('./models/regressor_model_type_{}_epoch_{}_batch_{}_minimax_{}'
                                      .format(model_type, epochs, batch_size, min_max), csv_path)

