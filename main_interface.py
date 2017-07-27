import functions
import random
import sys
from export_log import outputResults
#--------
import time
import timeout 
import math
import inspect

# positive_set is the set of positive examples, as a file
# negative_set is the set of negative examples, as a file
# percent_test is the portion of the imput sets that should be used as the test set
# The rest will be the training set.
# representation is the representation to use, as a string
# classifier is the classifier to use, as a string
# extra is other information that is used to specify the behavior of the classifier
def analogy_trial(positive_set, negative_set, percent_test, representation, classifier, extra={"sub_class":""}, timer=1000000000, comment=""):
    caller = inspect.stack()[1][3]
    start = time.time()
    # Read in the set of positive examples
    analogy_list = functions.get_list_re(positive_set)
    # Read in the set of negative examples
    non_analogy_list = functions.get_list_re(negative_set)
    # Randomly divide them into a training set and a test set
    samples = [(text, 'YES') for text in analogy_list] + [(text, 'NO') for text in non_analogy_list]
    extra = functions.set_extra(extra)
    # Run classifier, generate results based on the value passed in for representation
    beginTimer = time.time()
    train_data, train_labels, test_data, test_labels = functions.preprocess(samples, percent_test, caller)
    # Make sure the classifier runs within a set time
    try:
        score, matrix, precision, recall, f_measure = functions.classify(train_data, train_labels, test_data, test_labels, classifier, representation, extra, timer)
        
    # catch the timeout error
    except timeout.TimeoutError:
        print("Classifier timeout.")
        print("Output error in log.")
        algoTime = time.time()-beginTimer
        runTime = time.time()-start
        outputData = [positive_set, negative_set, percent_test, representation, classifier, extra, "", "", "", "", "", "", "", "Algorithm Timeout"]
        outputResults(outputData)
    
    else:
        algoTime = time.time()-beginTimer
        runTime = time.time()-start
        outputData = [positive_set, negative_set, percent_test, representation, classifier, extra, score, matrix, precision, recall, f_measure, runTime, algoTime, comment]
    
    # Store results
    outputResults(outputData)
    if caller != "test_main_interface_output":
        print("Successfully logged trial results")
    outputData = outputData[7:-3]
    outputData[1] = outputData[1].tolist()
    return outputData

if __name__ == '__main__':
    positive_set = 'test_extractions/bc_samples.txt'
    negative_set = 'test_extractions/bc_grounds.txt'
    #analogy_trial(positive_set, negative_set, .5, 'count', 'neural', {"sub_class":"nusvc", "stop_words":'english', "max_df":0.8, "activation":"tanh", "learning_rate":"adaptive"})
    analogy_trial(positive_set, negative_set, .5, "count", "naive", {"sub_class":""}, timer= 5)
    