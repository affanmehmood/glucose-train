import tensorflow as tf
import pandas as pd

import sequential_net_softmax_forsaken
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K


def custom_mse(alpha=0.01):
    def _mse(y_true, y_pred):
        error = y_true - y_pred
        sqr_error = K.square(error)
        mean_sqr_error = K.mean(sqr_error)
        sqrt_mean_sqr_error = K.sqrt(mean_sqr_error)
        return sqrt_mean_sqr_error + K.square(alpha)
    return _mse


class Block(tf.keras.Model):
    def __init__(self, depth):
        super(Block, self).__init__()
        self.depth = depth

        self.units = 64
        for i in range(0, depth):
            vars(self)[f'dense_{i}'] = tf.keras.layers.Dense(self.units, activation="relu")
            self.units *= 2

        self.dropout = tf.keras.layers.Dropout(0.1)

    def call(self, inputs):
        x = vars(self)['dense_0'](inputs)

        for i in range(1, self.depth):
            dense_layer = vars(self)[f'dense_{i}']
            x = dense_layer(x)

        dropout = self.dropout(x)
        return dropout


class GlucoNet(tf.keras.Model):
    def __init__(self):
        super(GlucoNet, self).__init__()
        self.batch_norm = tf.keras.layers.BatchNormalization()

        self.block_a = Block(8)
        self.block_b = Block(8)
        self.block_a_c = Block(8)
        self.block_b_c = Block(8)

        self.block_d = Block(8)
        self.block_e = Block(8)
        self.block_d_f = Block(8)
        self.block_e_f = Block(8)

        self.block_g = Block(8)
        self.block_h = Block(8)
        self.block_g_i = Block(8)
        self.block_h_i = Block(8)

        self.block_j = Block(8)
        self.block_k = Block(8)
        self.block_j_l = Block(8)
        self.block_k_l = Block(8)

        self.concat1 = tf.keras.layers.Concatenate(axis=1)
        self.concat2 = tf.keras.layers.Concatenate(axis=1)
        self.concat3 = tf.keras.layers.Concatenate(axis=1)
        self.concat4 = tf.keras.layers.Concatenate(axis=1)

        self.block_4_cat1 = Block(8)
        self.block_4_cat2 = Block(8)

        self.output_dense = tf.keras.layers.Dense(1, name="output_dense")

    def call(self, inputs):
        x = self.batch_norm(inputs)

        out_a = self.block_a(x)
        out_b = self.block_b(x)
        out_a_c = self.block_a_c(out_a)
        out_b_c = self.block_b_c(out_b)
        c1 = self.concat1([out_a_c, out_b_c])

        out_d = self.block_d(x)
        out_e = self.block_e(x)
        out_d_f = self.block_d_f(out_d)
        out_e_f = self.block_e_f(out_e)
        c2 = self.concat2([out_d_f, out_e_f])

        out_g = self.block_g(c1)
        out_h = self.block_h(c1)
        out_g_i = self.block_g_i(out_g)
        out_h_i = self.block_h_i(out_h)

        out_j = self.block_j(c2)
        out_k = self.block_k(c2)
        out_j_l = self.block_j_l(out_j)
        out_k_l = self.block_k_l(out_k)

        x = self.concat3([out_g_i, out_h_i])
        x1 = self.block_4_cat1(x)

        x = self.concat4([out_j_l, out_k_l])
        x2 = self.block_4_cat2(x)

        x = self.concat4([x1, x2])

        x = self.output_dense(x)
        return x


if __name__ == '__main__':
    data = pd.read_csv("../datasets/cleaned_enne.csv")

    train_df, eval_df, test_df = sequential_net.prepare_data_splits_from_dataframe(data)

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

    model = GlucoNet()
    batch_size = 64
    epochs = 10
    model.compile(optimizer="adam", loss=custom_mse(0.08),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    history = model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, validation_data=(val_X, val_Y),
                        verbose=True)
    plot_model(model, to_file='model.png')

    print(model.summary())
    plot_model(model, to_file='model.png')
    # model.save('../models/gluconet_model')
