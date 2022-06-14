from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
from preprocessing.utils import scale_data_std
import joblib
import numpy as np

min_max = False


def run_k_means(df):
    random_state = 0
    # Incorrect number of clusters
    k_means = KMeans(n_clusters=3, random_state=random_state, init="k-means++")
    y_pred = k_means.fit_predict(df)
    count_arr = np.bincount(y_pred)
    print(count_arr)
    print(set(y_pred))

    return k_means


if __name__ == "__main__":
    if min_max:
        data = pd.read_csv("datasets/cleaned_enne_minmax.csv")
    else:
        data = pd.read_csv("datasets/cleaned_enne.csv")
    # drop duplicates
    data.drop_duplicates(subset=['sex', 'age', 'height', 'weight',
                                 'heartrate', 'bpm', 'ibi',
                                 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2',
                                 's', 'sd1/sd2', 'breathingrate'], keep='first', inplace=True)
    # train_df, eval_df, test_df = prepare_data_splits_from_dataframe(data)
    train_df = data
    # split target from train
    train_X = data.loc[:, 'sex':'breathingrate']
    train_X.reset_index(drop=True, inplace=True)

    train_X, labels_true = make_blobs(
        n_samples=len(train_X.values), centers=train_X.values, cluster_std=0.5, random_state=0
    )

    # Normalize data
    train_X, scaler = scale_data_std(train_X)
    k_means = run_k_means(train_X)

    joblib.dump(scaler,
                './models/kmeans_minimax_{}_scaler.gz'
                .format(min_max))
    joblib.dump(k_means,
                './models/kmeans_minimax_{}.joblib'
                .format(min_max))
