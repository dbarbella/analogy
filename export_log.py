import time
import os
import csv

#creating variables for outputResults and whichClassifier
currentRunningFile = "analogy_svms"
OutputFileName = "Trial"
now = time.strftime("%c")
now  = now.replace(" ", "_")
now  = now.replace(":", "")

#output results to .csv file
def outputResults(testOutput):
    
    testOutput.insert(0, now)
     
    if os.path.exists("logs/exp_trials/"+OutputFileName+"_"+now+".csv"):
        with open(OutputFileName+"_"+now+".csv", 'a') as resultsFile:
            fileReader = csv.reader(resultsFile)
            fileWriter = csv.writer(resultsFile, quoting=csv.QUOTE_ALL)
            fileWriter.writerow(testOutput)
    else:
        with open("logs/exp_trials/"+OutputFileName+"_"+now+".csv", 'w') as resultsFile:
            fileWriter = csv.writer(resultsFile, quoting=csv.QUOTE_ALL)
            fileWriter.writerow(["datetime", "positive_set", "negative_set", "representation", "classifier", "extra", "score", "matrix", "precision", "recall", "f_measure", "Runtime(seconds)", "Algorithm_time(seconds)", "Comments"])
            fileWriter.writerow(testOutput)

