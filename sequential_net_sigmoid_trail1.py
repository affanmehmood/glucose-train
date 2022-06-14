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
from preprocessing.utils import prepare_dataset, draw_error_grids

model_type = 2  # 2 2 2
epochs = 15  # 10 15 20
batch_size = 32  # 32 32 64

csv_paths = [
    'datasets/-4580253642453236142-466387376852218172alltrialrecord.csv',
    # 'datasets/2173747691999486503022439312933411749allrecord_for_update_hyper_mode.csv',
    # 'datasets/-47054135960479185alltridata-05-19_05-20.csv',
    # 'datasets/-7791009196018876848alltridata-05-19_05-20.csv',
    # 'datasets/4792894170021136961alltrialdata-05-18_05-19-T2.csv'
]


def load_and_eval_model(model_path, csv_file_path):
    model = keras.models.load_model(model_path)

    # model.summary()

    data = prepare_dataset(csv_file_path)

    comp_data_X = data.loc[:, 'sex':'bloodGlucose']
    comp_data_Y = pd.DataFrame()
    comp_data_Y['hyper'] = pd.DataFrame(np.where(comp_data_X['bloodGlucose'] >= 10, 1, 0))
    # comp_data_Y['normal'] = pd.DataFrame(np.where(comp_data_X['bloodGlucose'] < 10, 1, 0))

    comp_data_X.drop('bloodGlucose', axis=1, inplace=True)

    # print('df_X.columns', comp_data_X.columns)
    # comp_data_X = (comp_data_X - comp_data_X.mean() / comp_data_X.std()) if (comp_data_X - comp_data_X.mean() /
    # comp_data_X.std()) else 43.662943
    scaler = joblib.load(model_path + '_scaler.gz')
    comp_data_X = scaler.transform(comp_data_X)

    predictions = model.predict(comp_data_X)
    predict_single = ['hyper' if pred[0] >= 0.5 else 'normal' for pred in predictions]

    og_df = pd.read_csv(csv_file_path)
    og_df = og_df[og_df['Glucometer Values'].notna()]
    og_df['Glucometer Original'] = og_df['Glucometer Values']
    # og_df['predict_values'] = predict_values
    og_df['predictions'] = predict_single
    og_df.to_excel(csv_file_path[:len(csv_file_path) - 4] + '-PREDS.xlsx')

    predictions = [1 if pred[0] >= 0.5 else 0 for pred in predictions]
    precision, recall, fscore, support = score(comp_data_Y['hyper'], predictions)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))


if __name__ == '__main__':
    for index, csv_path in enumerate(csv_paths):
        print('trail {}'.format(index))
        load_and_eval_model('./models/binary_type_{}_epoch_{}_batch_{}'.format(model_type, epochs, batch_size),
                            csv_path)
