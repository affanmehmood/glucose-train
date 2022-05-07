import tensorflow as tf
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv("datasets/cleaned_enne.csv")
    corr = df.corr()
    sb.heatmap(corr, cmap="Blues", annot=True)

    plt.show()
