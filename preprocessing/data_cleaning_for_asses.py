import csv
import json
import pandas as pd


new_data = []
tempDict = {}

count = 0
skipCount = 0
with open('../datasets/enne.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        # dataframe = pd.DataFrame(row)

        row[3] = row[3].replace("'", '"').replace('True', '"True"').replace('False', '"False"')
        row[8] = row[8].replace("'", '"').replace('True', '"True"').replace('False', '"False"')

        # print(row)
        if count == 0:
            print(row)
            count += 1
            continue

        profileDict = json.loads(row[3])
        dataDict = json.loads(row[8])

        # skip if HR == -1 or glucose = 0
        if ('features' in dataDict and dataDict['features'][0]['heartrate'] == -1) \
                or ('features' in dataDict and dataDict['features'][0]['bloodGlucose'] == float(0)):
            skipCount += 1
            continue
        if 'diabeticType' in profileDict and profileDict['diabeticType'] == 'type-2' and 'features' in dataDict:
            new_data.append([row[2], dataDict['features'][0]['heartrate'], row[15], row[11], row[13],
                             dataDict['features'][0]['bloodGlucose']])

            if row[2] in tempDict:
                minHR = dataDict['features'][0]['heartrate'] if tempDict[row[2]]['minHR'] > dataDict['features'][0][
                    'heartrate'] \
                    else tempDict[row[2]]['minHR']

                maxHR = dataDict['features'][0]['heartrate'] if tempDict[row[2]]['maxHR'] < dataDict['features'][0][
                    'heartrate'] \
                    else tempDict[row[2]]['maxHR']
                tempDict[row[2]] = {
                    'minHR': minHR,
                    'maxHR': maxHR
                }
            else:
                tempDict[row[2]] = {
                    'minHR': dataDict['features'][0]['heartrate'],
                    'maxHR': dataDict['features'][0]['heartrate']
                }

# calculate range
for user in new_data:
    hrRange = tempDict[user[0]]['maxHR'] - tempDict[user[0]]['minHR']
    user.insert(2, hrRange)
    del user[0]

pd.DataFrame(new_data).to_csv("../datasets/asses_config1.csv", index=False,
                              header=['HR', 'HRRange', 'pNN50', 'stdNN', 'rmssd', 'glucoseLevel'])


print('total records', len(new_data))
print('total rows skipped', skipCount)
