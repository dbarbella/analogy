import glob
import csv

path = "/logs/exp_trials/"

with open("logs/consolidated/"+"consolidated_log.csv", 'w') as resultsFile:
    fileWriter = csv.writer(resultsFile, quoting=csv.QUOTE_ALL)
    fileWriter.writerow(["datetime", "positive_set", "negative_set", "representation", "classifier", "extra", "score", "matrix", "precision", "recall", "f_measure", "Runtime(seconds)", "Algorithm_time(seconds)", "Comments"])
           
    for fileName in glob.glob(path):
        with open(fileName, 'r') as file:
            fileReader = csv.reader(file)
            next(fileReader, None)
            for row in fileReader:
                fileWriter.writerow(row)
    print("File successfully consolidated at Logs/consolidated.")
            
        
    
    
