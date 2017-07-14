import nltk
from analogy_strings import analogy_string_list
from sentence_parser import get_speech_tags
from personal import root
#------------------------
from nltk.classify import SklearnClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from nltk.classify import maxent
import sys
#------------------------
import random
import re
from boyer_moore import find_boyer_moore
#------------------------
import time
import os
import csv

# Implementation of the classifier to detect analogy.
# From http://www.nltk.org/book/ch06.html

analogy_file_name = "test_extractions/demo_analogy_samples.txt"
non_analogy_file_name = "test_extractions/demo_analogy_ground.txt"

def get_analogy_string(text, mode):
    # Returns a tuple of the first analogy indicator along with its speech tag in the text.
    # e.g. (('as', 'RB'), ('fast', 'RB'), ('as', 'IN'))
    result = []
    tokens = text.split()
    tagged_text = nltk.pos_tag(tokens)

    for pattern in analogy_string_list:
        match_index = find_boyer_moore(tokens, pattern)
        if match_index != -1:
            for i in range(len(pattern)):
                result.append(tagged_text[match_index + i])
            if (mode == "naive" or mode == "max_ent"):
                return {"analogy_indicator:" : tuple(result)}
            elif (mode == "svm"):
                return (result)

    if (mode == "naive" or mode == "max_ent"):
        return {"analogy_indicator:" : tuple(result)}
    elif (mode == "svm"):
        return (result)


def get_list(filename):
    # Returns all training data as a list
    # File should be formatted as a text line followed '>' in the next line
    # before a new text line.
    list = []
    file = open(filename, "r", encoding = "utf-8")
    for line in file.readlines():
        if line[0] != '>' and line != "\n":
            list.append(line)

    return list


analogy_list = get_list(analogy_file_name)
non_analogy_list = get_list(non_analogy_file_name)

# labeled data.
samples = [(text, 'YES') for text in analogy_list] + [(text, 'NO') for text in non_analogy_list]
random.shuffle(samples)

# verify that the classifier is implemented
mode = input("Please specify the type of classifer you want to use: svm/naive/max_ent? ")
if (mode != "svm") and (mode != "naive") and (mode != "max_ent"):
    sys.exit("This classifier has not been implemented yet.")

# divide data into training set and test set
feature_sets = [(get_analogy_string(text, mode), label) for (text, label) in samples]
train_set =  feature_sets[: 100]
test_set = feature_sets[100 :]


# change behavior according to the classifier
def whichClassifier(mode):    
    
    if mode == "naive":
        # train classifier with training set
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        # test classifier
        results = str(nltk.classify.accuracy(classifier, test_set))+ str(classifier.show_most_informative_features(1))
        print(results)
        # gather information for results file 
        testRunOutput = [currentRunningFile, now, mode, analogy_file_name, non_analogy_file_name, results]

    elif mode == "svm":
        # Preparing the data
        train_data = [text for (text, label) in train_set]
        train_labels = [label for (text, label) in train_set]
        test_data = [text for (text, label) in test_set]
        test_labels = [label for (text, label) in test_set]
        TfidfVect = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
        TfidfTrans = TfidfVect.fit_transform(train_data)
        TfidfTrans_test = TfidfVect.transform(test_data)
        # LinearSVC
        LinearSvc = SklearnClassifier(LinearSVC().fit(TfidfTrans, train_labels).score(TfidfTrans_test, test_labels))
        # SVC
        Svc = SklearnClassifier(SVC().fit(TfidfTrans, train_labels).score(TfidfTrans_test, test_labels))
        # NuSVC
        NuSvc = SklearnClassifier(NuSVC().fit(TfidfTrans, train_labels).score(TfidfTrans_test, test_labels))
        results = ("LinearSvc: "+ str(LinearSvc)+ "Svc: "+ str(Svc)+ "NuSvc: "+ str(NuSvc))
        print(results)
        #formatting output
        testRunOutput = [currentRunningFile, now, mode, analogy_file_name, non_analogy_file_name, results]

    elif mode == "max_ent":
        #running max_ent 
        max_ent = nltk.classify.MaxentClassifier.train(train_set, 'GIS', trace=0, max_iter=1000)
        results = ("Maximum Entropy: "+ str(nltk.classify.accuracy(max_ent, test_set))+ str(max_ent.show_most_informative_features(5)))
        print(results)
        #formatting output
        testRunOutput = [currentRunningFile, now, mode, analogy_file_name, non_analogy_file_name, results]
    
    return testRunOutput

#creating variables for outputResults and whichClassifier
currentRunningFile = "analogy_svms"
OutputFileName = "trial"
now = time.strftime("%c")
testOutput = whichClassifier(mode)

#output results to .csv file
def outputResults(testOutput):
     
    if os.path.exists(OutputFileName+"_"+now+".csv"):
        with open(OutputFileName+"_"+now+".csv", 'a') as resultsFile:
            fileReader = csv.reader(resultsFile)
            fileWriter = csv.writer(resultsFile, quoting=csv.QUOTE_ALL)
            fileWriter.writerow(testOutput)
    else:
        with open(OutputFileName+"_"+now+".csv", 'w') as resultsFile:
            fileWriter = csv.writer(resultsFile, quoting=csv.QUOTE_ALL)
            fileWriter.writerow(["File used", "Time and Date", "Classifier used", "Analogy file", "Not analogy file", "Results", "Comments"])
            fileWriter.writerow(testOutput)

outputResults(testOutput)



    