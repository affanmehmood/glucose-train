import simpful as sf
import pandas as pd
from custom_net import draw_error_grids

FS = sf.FuzzySystem()

data = pd.read_csv("datasets/cleaned_enne.csv")

# well use some of these features
inputs = ['age', 'height', 'weight',
          'heartrate', 'bpm', 'ibi',
          'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'hr_mad', 'sd1', 'sd2',
          's', 'sd1/sd2', 'breathingrate']

# Define a linguistic variable age.
T_1 = sf.FuzzySet(function=sf.Triangular_MF(a=12, b=19, c=26), term="young")
T_2 = sf.FuzzySet(function=sf.Trapezoidal_MF(a=22, b=34, c=44, d=50), term="adult")
T_3 = sf.FuzzySet(function=sf.Trapezoidal_MF(a=45, b=60, c=75, d=90), term="old")
FS.add_linguistic_variable("age", sf.LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[12, 90]))

# Define a linguistic variable height.
T_1 = sf.FuzzySet(function=sf.Triangular_MF(a=120, b=120, c=160), term="short")
T_2 = sf.FuzzySet(function=sf.Triangular_MF(a=150, b=165, c=180), term="average")
T_3 = sf.FuzzySet(function=sf.Trapezoidal_MF(a=170, b=187, c=200, d=220), term="tall")
FS.add_linguistic_variable("height", sf.LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[120, 220]))

# Define a linguistic variable weight.
T_1 = sf.FuzzySet(function=sf.Triangular_MF(a=30, b=30, c=55), term="light")
T_2 = sf.FuzzySet(function=sf.Triangular_MF(a=45, b=65, c=75), term="average")
T_3 = sf.FuzzySet(function=sf.Trapezoidal_MF(a=65, b=85, c=100, d=120), term="obese")
FS.add_linguistic_variable("weight", sf.LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[30, 120]))


# data[data['bloodGlucose'] < 4]['rmssd'].mean()
def get_range_bound_point(dataframe, key, upper_bound, lower_bound):
    return dataframe.loc[(dataframe['bloodGlucose'] < upper_bound) & (dataframe['bloodGlucose'] > lower_bound)][
        key].mean()


def get_range_point(dataframe, key, bound):
    return dataframe.loc[(dataframe['bloodGlucose'] < bound)][key].mean()


# Define a linguistic variable rmssd.
col_key = 'rmssd'
T_1 = sf.FuzzySet(function=sf.Trapezoidal_MF(a=data[data['bloodGlucose'] > 15][col_key].mean(),
                                             b=get_range_bound_point(data, col_key, 15, 14),
                                             c=get_range_bound_point(data, col_key, 14, 10),
                                             d=get_range_bound_point(data, col_key, 10, 8)), term="low")
T_2 = sf.FuzzySet(function=sf.Triangular_MF(a=get_range_bound_point(data, col_key, 8, 7),
                                            b=get_range_bound_point(data, col_key, 8.5, 6.8),
                                            c=get_range_bound_point(data, col_key, 6.5, 5)), term="average")
T_3 = sf.FuzzySet(function=sf.Trapezoidal_MF(a=get_range_bound_point(data, col_key, 5, 4),
                                             b=get_range_bound_point(data, col_key, 5.5, 4),
                                             c=get_range_bound_point(data, col_key, 4, 3.5),
                                             d=get_range_point(data, col_key, 3)), term="high")

FS.add_linguistic_variable(col_key, sf.LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[
    get_range_point(data, col_key, 14), get_range_point(data, col_key, 3)]))

# Define a linguistic variable rmssd.
col_key = 'heartrate'

T_1 = sf.FuzzySet(function=sf.Trapezoidal_MF(a=75,
                                             b=76.8,
                                             c=77.3,
                                             d=77.5), term="slow")
T_2 = sf.FuzzySet(function=sf.Triangular_MF(a=77,
                                            b=77.8,
                                            c=82.6), term="average")
T_3 = sf.FuzzySet(function=sf.Triangular_MF(a=82,
                                            b=85,
                                            c=87), term="fast")

FS.add_linguistic_variable(col_key, sf.LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[70, 95]))

# Define a linguistic variable glucose.
T_1 = sf.FuzzySet(function=sf.Triangular_MF(a=3, b=3, c=5.5), term="low")
T_2 = sf.FuzzySet(function=sf.Triangular_MF(a=5.5, b=7, c=8), term="medium")
T_3 = sf.FuzzySet(function=sf.Trapezoidal_MF(a=8, b=11, c=14, d=16), term="high")
FS.add_linguistic_variable("bloodGlucose", sf.LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[3, 16]))

FS.add_rules([
    "IF (age IS young) OR (weight IS light) OR (height IS tall) OR (rmssd IS high) OR (heartrate IS slow) THEN ("
    "bloodGlucose IS low)",
    "IF (age IS adult) OR (weight IS average) OR (height IS average) OR (rmssd IS average) OR (heartrate IS average) "
    "THEN (bloodGlucose IS medium)",
    "IF (age IS old) OR (weight IS obese) OR (height IS short) OR (rmssd IS low) OR (heartrate IS fast) THEN ("
    "bloodGlucose IS high) "
])

allrecord_df = pd.read_csv('./datasets/allrecord_cleaned.csv')

actual_glucose = []
predicted_glucose = []
for index, row in allrecord_df.iterrows():
    FS.set_variable("age", row['age'])
    FS.set_variable("height", row['height'])
    FS.set_variable("weight", row['weight'])
    FS.set_variable("rmssd", row['rmssd'])
    FS.set_variable("heartrate", row['heartrate'])

    tip = FS.inference()

    actual_glucose.append(row['bloodGlucose'])
    predicted_glucose.append(tip['bloodGlucose'])


new_df = pd.DataFrame({'actual': actual_glucose, 'predicted': predicted_glucose})
new_df.to_excel('./datasets/fuzzy_predictions.xlsx')
draw_error_grids(new_df['actual'], new_df['predicted'], 'fuzzy-match')
