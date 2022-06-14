import regression_model as regressor
from preprocessing.utils import scale_data_std
import joblib
import pandas as pd
import tensorflow as tf
import pickle
from preprocessing.utils import prepare_dataset, draw_error_grids, prepare_data_splits_from_dataframe

min_max = False


def train_rf():
    if min_max:
        data = pd.read_csv("datasets/cleaned_enne_minmax.csv")
    else:
        data = pd.read_csv("datasets/cleaned_enne.csv")
    # split target from train
    train_X = data.loc[:, 'sex':'breathingrate']
    train_Y = data.loc[:, 'bloodGlucose':'bloodGlucose']
    train_X.reset_index(drop=True, inplace=True)
    train_Y.reset_index(drop=True, inplace=True)

    # split target from eval
    # val_X = eval_df.loc[:, 'sex':'breathingrate']
    # val_Y = eval_df.loc[:, 'bloodGlucose':'bloodGlucose']
    # val_X.reset_index(drop=True, inplace=True)
    # val_Y.reset_index(drop=True, inplace=True)
    #
    # # split target from test
    # test_X = test_df.loc[:, 'sex':'breathingrate']
    # test_Y = test_df.loc[:, 'bloodGlucose':'bloodGlucose']
    # test_X.reset_index(drop=True, inplace=True)
    # test_Y.reset_index(drop=True, inplace=True)

    # Normalize data
    train_X, scaler = scale_data_std(train_X)

    rf_model = regressor.rf_fit(train_X, train_Y)
    with open('./models/rf_model_minimax_{}.pkl'.format(min_max), 'wb') as fid:
        pickle.dump(rf_model, fid)

    joblib.dump(scaler,
                './models/rf_model_minimax_{}_scaler.gz'
                .format(min_max))


def load_and_eval_model_trail_new(model, scalar, csv_file_path):
    data = prepare_dataset(csv_file_path, min_max=min_max)
    comp_data_X = data.loc[:, 'sex':'bloodGlucose']
    comp_data_Y = pd.DataFrame()
    comp_data_Y['bloodGlucose'] = comp_data_X['bloodGlucose']
    comp_data_X.drop('bloodGlucose', axis=1, inplace=True)

    comp_data_X = scalar.transform(comp_data_X)

    predictions = model.predict(comp_data_X)

    predictions = [pred for pred in predictions]
    ground_truth = [gt[0] for gt in comp_data_Y.values]
    # print(list(zip(ground_truth, predictions)))
    mae = tf.keras.losses.MeanAbsoluteError()
    print('Mean absolute error on trail 3:', mae(ground_truth, predictions).numpy())

    og_df = pd.read_csv(csv_file_path)
    og_df = og_df[og_df['Glucometer Values'].notna()]
    og_df['Glucometer Original'] = og_df['Glucometer Values']
    og_df['Predictions'] = predictions
    og_df.to_excel(csv_file_path[:len(csv_file_path) - 4] + 'Reg-RF-PREDS.xlsx')

    draw_error_grids(og_df['Glucometer Original'], og_df['Predictions'],
                     'rf_model_min_max_{}'.format(min_max))


if __name__ == '__main__':
    train_rf()
    rf_model = pickle.load(open('./models/rf_model_minimax_{}.pkl'.format(min_max), 'rb'))
    scaler = joblib.load('./models/rf_model_minimax_{}'.format(min_max) + '_scaler.gz')
    csv_paths = [
        # 'datasets/5979403437824095900alltrialrecord_update.csv',
        # 'datasets/-4580253642453236142-466387376852218172alltrialrecord.csv',
        'datasets/-8845707664459066312alltrialdata-05-16_05-18.csv'
    ]
    for csv_path in csv_paths:
        load_and_eval_model_trail_new(model=rf_model, scalar=scaler, csv_file_path=csv_path)
