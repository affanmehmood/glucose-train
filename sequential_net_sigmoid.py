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
from preprocessing.utils import scale_data_std
import pickle
import joblib

model_type = 3
epochs = 2000
batch_size = 32


def get_model_1(inputs):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(len(inputs),), name="input"),
        keras.layers.Dense(32, activation="relu", name="dense_1"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(64, activation="relu", name="dense_2"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(128, name="dense_3"),
        keras.layers.Dropout(0.1),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(256, name="dense_4"),
        keras.layers.Dropout(0.2),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(512, name="dense_5"),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(512, name="dense_6"),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(256, name="dense_7"),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(128, name="dense_8"),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(64, activation="relu", name="dense_9"),
        keras.layers.Dense(32, activation="relu", name="dense_10"),
        keras.layers.Dense(1, activation='sigmoid', name="output"),
    ], name="model")

    return model


# client has this
def get_model_2(inputs):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(len(inputs),), name="input"),
        keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_1"),
        keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_2"),
        keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_3"),
        keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_4"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_5"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_6"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_7"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_8"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_9"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_10"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(4096, kernel_initializer='normal', activation="relu", name="dense_11"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_12"),
        # keras.layers.Dropout(0.3),
        keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_13"),
        keras.layers.Dropout(0.3),
        # keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_14"),
        # keras.layers.Dropout(0.3),
        # keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_15"),
        # keras.layers.Dropout(0.3),
        # keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_16"),
        # keras.layers.Dropout(0.3),
        # keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_17"),
        # keras.layers.Dropout(0.3),
        # keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_18"),
        # keras.layers.Dropout(0.2),
        # keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_19"),
        # keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_20"),
        # keras.layers.Dense(512, kernel_initializer='normal', activation="relu", name="dense_27"),
        keras.layers.Dense(1, activation='sigmoid', name="output"),
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
        keras.layers.Dense(1, activation='sigmoid', name="output"),
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
        # keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_5"),
        # keras.layers.Dropout(0.3),
        # keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_6"),
        # keras.layers.Dropout(0.3),
        # keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_7"),
        # keras.layers.Dropout(0.3),
        keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_8"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_10"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(4096, kernel_initializer='normal', activation="relu", name="dense_11"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(4096, kernel_initializer='normal', activation="relu", name="dense_12"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(4096, kernel_initializer='normal', activation="relu", name="dense_13"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_14"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_15"),
        keras.layers.Dropout(0.3),
        # keras.layers.Dense(2048, kernel_initializer='normal', activation="relu", name="dense_16"),
        # keras.layers.Dropout(0.3),
        # keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_17"),
        # keras.layers.Dropout(0.3),
        # keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_18"),
        # keras.layers.Dropout(0.2),
        keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_19"),
        keras.layers.Dense(1024, kernel_initializer='normal', activation="relu", name="dense_20"),
        keras.layers.Dense(512, kernel_initializer='normal', activation="relu", name="dense_27"),
        keras.layers.Dense(1, activation='sigmoid', name="output"),
    ], name="model")
    return model


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
    train_X.drop('bloodGlucose', axis=1, inplace=True)

    train_X.reset_index(drop=True, inplace=True)
    train_Y.reset_index(drop=True, inplace=True)

    # Normalize data
    train_X, scaler = scale_data_std(train_X)

    # Select labels for inputs and outputs.
    inputs = ['sex', 'age', 'height', 'weight',
              'heartrate', 'bpm', 'ibi',
              'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2',
              's', 'sd1/sd2', 'breathingrate']
    outputs = ["hyper"]

    if model_type == 1:
        model = get_model_1(inputs)
    elif model_type == 2:
        model = get_model_2(inputs)
    elif model_type == 3:
        model = get_model_3(inputs)
    elif model_type == 4:
        model = get_model_4(inputs)

    print('training without a validation set, batch size {}, epochs {}'
          .format(batch_size, epochs))
    model.compile(optimizer="adam", loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs,
              verbose=True)

    predictions = model.predict(train_X)

    predictions = [1 if pred[0] >= 0.5 else 0 for pred in predictions]

    precision, recall, fscore, support = score(train_Y['hyper'], predictions)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))

    # save the model
    model.save('./models/binary_type_{}_epoch_{}_batch_{}'.format(model_type, epochs, batch_size))
    joblib.dump(scaler, './models/binary_type_{}_epoch_{}_batch_{}_scaler.gz'.format(model_type, epochs, batch_size))


def load_and_eval_model(model_path):
    model = keras.models.load_model(model_path)

    # model.summary()

    data = pd.read_csv("datasets/test_cleaned.csv")
    comp_data_X = data.loc[:, 'sex':'bloodGlucose']
    comp_data_Y = pd.DataFrame()
    comp_data_Y['hyper'] = pd.DataFrame(np.where(comp_data_X['bloodGlucose'] >= 10, 1, 0))
    # comp_data_Y['normal'] = pd.DataFrame(np.where(comp_data_X['bloodGlucose'] < 10, 1, 0))
    comp_data_X.drop('bloodGlucose', axis=1, inplace=True)

    # normalize
    # comp_data_X = (comp_data_X - comp_data_X.mean()) / comp_data_X.std()
    scaler = joblib.load(model_path + '_scaler.gz')
    comp_data_X = scaler.transform(comp_data_X)

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
    # print(predictions, comp_data_Y)
    predictions = [1 if pred[0] >= 0.5 else 0 for pred in predictions]
    # print('f1 score:', f1_score(comp_data_Y['hyper'], predictions, average="macro"))
    # print('precision:', precision_score(comp_data_Y['hyper'], predictions, average="macro"))
    # print('recall:', recall_score(comp_data_Y['hyper'], predictions, average="macro"))
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
    main_exp()
    load_and_eval_model('./models/binary_type_{}_epoch_{}_batch_{}'.format(model_type, epochs, batch_size))
