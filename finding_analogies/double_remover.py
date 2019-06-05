import pandas as pd
from sys import argv
from fnmatch import filter
from os import listdir
from pick_random import get_random_analogy
"""
removes all rows with repeating ids in a directory
"""
csv_dir = argv[1]

if csv_dir[-1] != "/":
    csv_dir += "/"

csv_file = filter(listdir(csv_dir), '*.csv')
ids = set()

for file in csv_file:
    path = csv_dir + file
    data = pd.read_csv(path)
    for i in data["name"]:
        if i not in ids:
            ids.add(i)
            print(i, "worked")
        else:
            while i in ids:
                #data.drop(i, axis = 1)
                print(i, "didnt work")
                data = data[data.name != i]
                analogy = get_random_analogy()
                data.append(analogy)
            i = analogy["name"]
            ids.add(i)
            data.to_csv(path, encoding='utf-8', index=False)
        
        