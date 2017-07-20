import glob
import csv
import pandas as pd

path = "logs/"
allFiles = glob.glob("logs/exp_trials/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)

frame['datetime'] = pd.to_datetime(frame['datetime'])
frame.sort_values(by='datetime')
frame.to_csv('logs/consolidated/consolidated_log.csv', encoding='utf-8', quoting=csv.QUOTE_ALL)      
print("Successfully consolidated the file")
