from pytorch_tabnet.tab_model import TabNetRegressor
from ml_tech import draw_error_grids, prepare_data_splits_from_dataframe
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv("./datasets/cleaned_enne.csv")

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

    saving_path_name = "./models/tabnet_model_allrecord/"
    saved_filepath = './models/tabnet_model_allrecord.zip'

    clf = TabNetRegressor()  # TabNetRegressor()
    # clf.fit(
    #     train_X.values, train_Y.values,
    #     eval_set=[(val_X.values, val_Y.values)],
    #     max_epochs=100, patience=20,
    #     batch_size=16, virtual_batch_size=8
    # )

    clf.load_model(saved_filepath)

    allrecord = pd.read_csv("./datasets/allrecord_cleaned.csv")

    allrecord_X = allrecord.loc[:, 'sex':'breathingrate']
    allrecord_Y = allrecord.loc[:, 'bloodGlucose':'bloodGlucose']

    predictions = clf.predict(allrecord_X.values)
    predictions = [pred[0] for pred in predictions]

    allrecord['predicted'] = predictions

    allrecord.to_excel('./datasets/allrecord_predicted.xls')
    # allrecord_X['predict'] = predictions
    # draw_error_grids(test_Y['bloodGlucose'], test_Y['predict'], 'tabnet-allrecord-train')

    # clf.save_model(saving_path_name)
