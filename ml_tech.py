import methcomp
import numpy as np
import matplotlib.pyplot as plt
import regression_model as regressor

import pandas as pd
import pickle


def draw_error_grids(Y_test_users, y_pred_users, title):
    plt.figure(figsize=(12, 12))
    plot = methcomp.glucose.parkes(type=2, reference=np.array(Y_test_users.values * 18), units='mg/dl',
                                   test=np.array(y_pred_users * 18), color_points='black')
    plot.figure.savefig('./saved_figs/Parkes error plot - ' + title + '.png')

    plt.figure(figsize=(12, 12))
    plot = methcomp.glucose.clarke(reference=np.array(Y_test_users.values * 18), units='mg/dl',
                                   test=np.array(y_pred_users * 18), color_points='black')
    plot.figure.savefig('./saved_figs/Clarkes error plot - ' + title + '.png')


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


if __name__ == '__main__':
    data = pd.read_csv("datasets/cleaned_enne.csv")

    train_df, eval_df, test_df = prepare_data_splits_from_dataframe(data)

    # split target from train
    train_X = train_df.loc[:, 'sex':'breathingrate']
    train_Y = train_df.loc[:, 'bloodGlucose':'bloodGlucose']
    train_X.reset_index(drop=True, inplace=True)
    train_Y.reset_index(drop=True, inplace=True)

    # split target from eval
    val_X = eval_df.loc[:, 'sex':'breathingrate']
    val_Y = eval_df.loc[:, 'bloodGlucose':'bloodGlucose']
    val_X.reset_index(drop=True, inplace=True)
    val_Y.reset_index(drop=True, inplace=True)

    # split target from test
    test_X = test_df.loc[:, 'sex':'breathingrate']
    test_Y = test_df.loc[:, 'bloodGlucose':'bloodGlucose']
    test_X.reset_index(drop=True, inplace=True)
    test_Y.reset_index(drop=True, inplace=True)

    # Normalize data
    train_X = (train_X - train_X.mean()) / train_X.std()
    test_X = (test_X - test_X.mean()) / test_X.std()
    val_X = (val_X - val_X.mean()) / val_X.std()

    rf_model = regressor.gb_fit(train_X, train_Y)
    # with open('saved_ml_model/rf_model.pkl', 'wb') as fid:
    #     pickle.dump(rf_model, fid)

    # rf_model = pickle.load(open('saved_ml_model/rf_model.pkl', 'rb'))

    y_pred_users = rf_model.predict(test_X)

    df = pd.DataFrame()
    df['userkey'] = test_df['userkey']
    df['bloodGlucose'] = test_Y['bloodGlucose']
    df['prediction'] = y_pred_users

    draw_error_grids(test_Y, y_pred_users, 'rf')
