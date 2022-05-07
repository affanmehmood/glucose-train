import pandas as pd

data = pd.read_csv("datasets/enne.csv")

print(len(data))
data.drop_duplicates(subset=['userkey', 'bpm', 'ibi',
                           'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2',
                           's', 'sd1/sd2', 'breathingrate'], keep='first', inplace=True)
print(len(data))

