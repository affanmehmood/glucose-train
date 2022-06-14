from tensorflow import keras
import pandas as pd
import tensorflow as tf
import methcomp
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score


# # Select columns and remove rows with missing values.
# columns = ["HR", "HRRange", "pNN50", "stdNN", "rmssd", "glucoseLevel"]
#
# # droping -1s
# data = data[data != -1]
# data = data[columns].dropna(axis=0)
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


epochs = 100
batch_size = 64


def main_exp():
    data = pd.read_csv("datasets/cleaned_enne.csv")

    # train_df, eval_df, test_df = prepare_data_splits_from_dataframe(data)
    train_df = data
    # split target from train
    train_X = train_df.loc[:, 'sex':'bloodGlucose']
    train_Y = pd.DataFrame()
    train_Y['hyper'] = pd.DataFrame(np.where(train_X['bloodGlucose'] >= 10, 1, 0))
    sm = SMOTE(random_state=42, sampling_strategy='all')
    train_X, train_Y = sm.fit_resample(train_X, train_Y)
    train_Y['normal'] = pd.DataFrame(np.where(train_Y['hyper'] == 1, 0, 1))
    train_X.drop('bloodGlucose', axis=1, inplace=True)
    # print(train_Y.hyper.value_counts())
    # exit()
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
    train_X = (train_X - train_X.mean()) / train_X.std()
    # test_X = (test_X - test_X.mean()) / test_X.std()
    # val_X = (val_X - val_X.mean()) / val_X.std()

    # Select labels for inputs and outputs.
    inputs = ['sex', 'age', 'height', 'weight',
              'heartrate', 'bpm', 'ibi',
              'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2',
              's', 'sd1/sd2', 'breathingrate']
    outputs = ["hyper"]

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
        keras.layers.Dense(512, activation="sigmoid", name="dense_5"),
        keras.layers.Dense(512, activation="sigmoid", name="dense_6"),
        keras.layers.Dense(256, activation="sigmoid", name="dense_7"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu", name="dense_8"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation="relu", name="dense_9"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(32, activation="relu", name="dense_10"),
        keras.layers.Dense(2, activation='softmax', name="output"),
    ], name="model")

    print('training with a validation set, batch size {}, epochs {}'
          .format(batch_size, epochs))
    model.compile(optimizer="adam", loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    print(train_X.head(), train_Y.head())
    history = model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs,
                        verbose=True)

    predictions = model.predict(train_X)
    new_preds = []
    for pred, og in zip(predictions, train_Y.values):
        new_preds.append([og, pred])
    count_0 = 0
    count_0_T = 0
    count_1 = 0
    count_1_T = 0
    for new_pred in new_preds:
        orignal_value = new_pred[0][0]
        predicted_value = 1 if new_pred[1][0] >= 0.5 else 0
        if orignal_value != predicted_value and orignal_value == 0:
            count_0 += 1
        elif orignal_value != predicted_value and orignal_value == 1:
            count_1 += 1
        elif orignal_value == predicted_value and orignal_value == 0:
            count_0_T += 1
        elif orignal_value == predicted_value and orignal_value == 1:
            count_1_T += 1
    print('total predictions', len(new_preds))
    print('miss classified hyper', count_1, 'miss classified normal', count_0)
    print('correctly classified hyper', count_1_T, 'correctly classified normal', count_0_T)

    # save the model
    model.save('./models/b_classify_model_epoch_{}_batch_{}'.format(epochs, batch_size))

    # training loss graph
    # loss = history.history['loss']
    # epochs = range(1, len(loss) + 1)
    # plt.plot(epochs, loss, 'ro', linewidth=2, markersize=4, label='Training loss')
    # plt.savefig("./saved_figs/train_loss.png")

    # validation loss graph
    # val_loss = history.history['val_loss']
    # val_epochs = range(1, len(val_loss) + 1)
    # plt.plot(val_epochs, val_loss, 'ro', linewidth=2, markersize=4, label='Validation loss')
    # plt.savefig("./saved_figs/val_loss.png")


def load_and_eval_model(model_path):
    model = keras.models.load_model(model_path)
    # model summary and results
    # model.summary()

    # results = model.evaluate(test_X, test_Y)
    # print(results)
    data = pd.read_csv("datasets/test_cleaned.csv")
    comp_data_X = data.loc[:, 'sex':'bloodGlucose']
    comp_data_Y = pd.DataFrame()
    comp_data_Y['hyper'] = pd.DataFrame(np.where(comp_data_X['bloodGlucose'] >= 10, 1, 0))
    comp_data_Y['normal'] = pd.DataFrame(np.where(comp_data_X['bloodGlucose'] < 10, 1, 0))
    comp_data_X.drop('bloodGlucose', axis=1, inplace=True)
    comp_data_X = (comp_data_X - comp_data_X.mean()) / comp_data_X.std()
    predictions = model.predict(comp_data_X)

    # predictions = [pred[0] for pred in predictions]
    new_preds = []
    for pred, og in zip(predictions, comp_data_Y.values):
        # new_preds.append([og, pred])
        new_preds.append([og[0], pred[0]])
    # print(new_preds)
    # count_0 = 0
    # count_0_T = 0
    # count_1 = 0
    # count_1_T = 0
    # for new_pred in new_preds:
    #     orignal_value = new_pred[0][0]
    #     predicted_value = 1 if new_pred[1][0] >= 0.5 else 0
    #     if orignal_value != predicted_value and orignal_value == 0:
    #         count_0 += 1
    #     elif orignal_value != predicted_value and orignal_value == 1:
    #         count_1 += 1
    #     elif orignal_value == predicted_value and orignal_value == 0:
    #         count_0_T += 1
    #     elif orignal_value == predicted_value and orignal_value == 1:
    #         count_1_T += 1
    # print('total predictions', len(new_preds))
    # print('miss classified hyper', count_1, 'miss classified normal', count_0)
    # print('correctly classified hyper', count_1_T, 'correctly classified normal', count_0_T)
    # print('hyper', comp_data_Y['hyper'])
    predictions = [1 if pred[0] >= 0.5 else 0 for pred in predictions]
    precision, recall, fscore, support = score(comp_data_Y['hyper'], predictions)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    exit()
    # print('test_X[:10]', test_Y[:10])
    # print('predictions[:10]', predictions[:10])

    # comp_data_Y['predict'] = predictions

    # draw_error_grids(comp_data_Y['bloodGlucose'], comp_data_Y['predict'], 'dl-allrecord')

    # test_X.to_csv("predictions.csv", index=True)

    print('saved predictions')


if __name__ == '__main__':
    # main_exp()
    load_and_eval_model('./models/b_classify_model_epoch_{}_batch_{}'.format(epochs, batch_size))
