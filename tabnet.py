from pytorch_tabnet.tab_model import TabNetRegressor
from ml_tech import draw_error_grids
import pandas as pd
from torch import nn, from_numpy
from preprocessing.utils import prepare_dataset, draw_error_grids

saving_path_name = "./models/regressor_tabnet_enne"
saved_filepath = './models/regressor_tabnet_enne.zip'


def train():
    data = pd.read_csv("./datasets/cleaned_enne.csv")
    train_df = data

    train_X = train_df.loc[:, 'sex':'breathingrate']
    train_Y = train_df.loc[:, 'bloodGlucose':'bloodGlucose']
    train_X.reset_index(drop=True, inplace=True)
    train_Y.reset_index(drop=True, inplace=True)

    clf = TabNetRegressor(n_d=16, n_a=16, n_steps=9)
    clf.fit(
        train_X.values, train_Y.values,
        max_epochs=1000, patience=100,
        batch_size=256, virtual_batch_size=32,
        loss_fn=nn.L1Loss()
    )

    clf.save_model(saving_path_name)


def trail_3():
    clf = TabNetRegressor()
    clf.load_model(saved_filepath)

    csv_file_path = 'datasets/4792894170021136961alltrialdata-05-18_05-19-T2.csv'
    data = prepare_dataset(csv_file_path)

    test_data_X = data.loc[:, 'sex':'breathingrate']
    test_data_Y = data.loc[:, 'bloodGlucose':'bloodGlucose']
    test_data_X.reset_index(drop=True, inplace=True)
    test_data_Y.reset_index(drop=True, inplace=True)

    predictions = clf.predict(test_data_X.values)

    predictions = [pred[0] for pred in predictions]

    og_df = pd.read_csv(csv_file_path)
    og_df = og_df[og_df['Glucometer Values'].notna()]
    og_df['Glucometer Glucose'] = og_df['Glucometer Values']
    og_df['predicted'] = predictions
    # og_df.to_excel('./datasets/tabnet-trail-3_predicted.xls')

    loss = nn.L1Loss()
    output = loss(from_numpy(test_data_Y['bloodGlucose'].values), from_numpy(og_df['predicted'].values)).item()

    print('MAE for trail 3 :', output)
    draw_error_grids(test_data_Y['bloodGlucose'], og_df['predicted'], 'tabnet-trail-3-train')


def trail_4():
    clf = TabNetRegressor()
    clf.load_model(saved_filepath)

    csv_file_path = 'datasets/-7791009196018876848alltridata-05-19_05-20.csv'
    data = prepare_dataset(csv_file_path)

    test_data_X = data.loc[:, 'sex':'breathingrate']
    test_data_Y = data.loc[:, 'bloodGlucose':'bloodGlucose']
    test_data_X.reset_index(drop=True, inplace=True)
    test_data_Y.reset_index(drop=True, inplace=True)

    predictions = clf.predict(test_data_X.values)

    predictions = [pred[0] for pred in predictions]

    og_df = pd.read_csv(csv_file_path)
    og_df = og_df[og_df['Glucometer Values'].notna()]
    og_df['Glucometer Glucose'] = og_df['Glucometer Values']
    og_df['predicted'] = predictions
    # og_df.to_excel('./datasets/tabnet-trail-4_predicted.xls')

    loss = nn.L1Loss()
    output = loss(from_numpy(test_data_Y['bloodGlucose'].values), from_numpy(og_df['predicted'].values)).item()
    print('MAE for trail 4 :', output)
    draw_error_grids(test_data_Y['bloodGlucose'], og_df['predicted'], 'tabnet-trail-4-train')


def evaluate():
    trail_3()
    trail_4()


if __name__ == '__main__':
    train()
    evaluate()
