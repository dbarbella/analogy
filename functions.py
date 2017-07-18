import nltk
from analogy_strings import analogy_string_list
from sentence_parser import get_speech_tags
from personal import root
#------------------------
from nltk.classify import SklearnClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import sys
#------------------------
import random
import re
from boyer_moore import find_boyer_moore

def get_list(filename):
    # Returns all training data as a list
    # File should be formatted as a text line followed '>' in the next line
    # before a new text line.
    list = []
    file = open(filename, "r", encoding = "utf-8")
    for line in file.readlines():
        if line[0] != '>' and line != "\n":
            final = line.split("]\",")[-1].split(".")[0].split("?")[0].split("!")[0]
            list.append(final)

    return list

# Returns all training data as a list which contains only the text(removing the source, paragraph #, sentence #, ratings
def get_list_re(filename):
    list = []
    file = open(filename, "r", encoding = "utf-8")
    for line in file.readlines():
        if line[0] != '>' and line != "\n":
            l = line.split("]\",")[-1]
            final = re.sub("[^a-zA-Z]"," ", l)
            list.append(final)

    return list

# preprocess the data so it can be used by the classifiers
def preprocess(samples, percent_test):
    num_samples = len[samples]
    random.shuffle(samples)
    cutoff = int((1.0 - percent_test) * num_samples)
    # create a train set and a test/development set
    feature_sets = [(text, label) for (text, label) in samples]
    train_set =  feature_sets[:cutoff]
    test_set = feature_sets[cutoff:]
    # separate the training data and the training labels
    train_data = [text for (text, label) in train_set]
    train_labels = [label for (text, label) in train_set]
    # separate the test data and the test labels
    test_data = [text for (text, label) in test_set]
    test_labels = [label for (text, label) in test_set]
    return(train_data, train_labels, test_data, test_labels)
    
# Transform the data so it can be represented using tfidf    
def tfidf(train_data, test_data):
    TfidfVect = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    TfidfTrans = TfidfVect.fit_transform(train_data)
    TfidfTrans_test = TfidfVect.transform(test_data)
    return(TfidfTrans, TfidfTrans_test)

# Transform the data so it can be represented using Count Vectorizer
def countvect(train_data, test_data):
    CountVect = CountVectorizer(lowercase=False)
    CountTrans = CountVect.fit_transform(train_data)
    CountTest = CountVect.transform(test_data)
    return(CountTrans, CountTest)

# Transform the data so it can be represented using Hashing Vectorizer
def hashing(train_data, test_data):
    HashVect = HashingVectorizer(lowercase=False)
    HashTrans = HashVect.fit_transform(train_data)
    HashTest = HashVect.transform(test_data)
    return(HashTrans, HashTest)

# Implementetion of the fmeasure metric, which calculates the precision, recall and f1measure given a confusion matrix
def fmeasure(matrix):
    precision = matrix[0][0] / (matrix[0][1] + matrix[0][0])
    recall = matrix[0][0] / (matrix[1][0] + matrix[0][0])
    f_measure = (2 * precision * recall) / (precision + recall)
    return(precision, recall, f_measure)

# A function which classifies data using different SVMs    
def svm(train_data, train_labels, test_data, test_labels, representation, extra=[]):
    # if the classifier not specified, use SVC as the default one
    if extra == "":
        # using tfidf representation
        if representation == "tfidf":
            TfidfTrans, TfidfTrans_test = tfidf(train_data, test_data)
            Svc_tf = SVC().fit(TfidfTrans, train_labels)
            test_predict_Svc_tf = Svc_tf.predict(TfidfTrans_test)
            score = Svc_tf.score(TfidfTrans_test, test_labels)
            matrix = confusion_matrix(test_labels,test_predict_Svc_tf,labels=["YES", "NO"])
            precision, recall, f_measure = fmeasure(matrix)
            return(score, matrix, precision, recall, f_measure)
        # using Count Vectorizer representation
        elif representation == "count":
            CountTrans, CountTest = countvect(train_data, test_data)
            Svc_count = SVC().fit(CountTrans, train_labels)
            test_predict_Svc_count = Svc_count.predict(CountTest)
            score = Svc_tf.score(TfidfTrans_test, test_labels)
            matrix = confusion_matrix(test_labels,test_predict_Svc_hash,labels=["YES", "NO"])
            precision, recall, f_measure = fmeasure(matrix)
            return(score, matrix, precision, recall, f_measure)
        # using Hashing Vectorizer representation
        elif representation == "hash":
            HashTrans, HashTest = hashing(train_data, test_data)
            Svc_hash = SVC().fit(HashTrans, train_labels)
            test_predict_Svc_hash = Svc_hash.predict(HashTest)
            score = Svc_hash.score(HashTest, test_labels)
            matrix = confusion_matrix(test_labels,test_predict_Svc_hash,labels=["YES", "NO"])
            precision, recall, f_measure = fmeasure(matrix)
            return(score, matrix, precision, recall, f_measure)
        else:
            sys.exit("This representation has not been implemented yet.")
    # if the classifier is specified: linear/nusvc
    elif extra == "linear":
        if representation == "tfidf":
            TfidfTrans, TfidfTrans_test = tfidf(train_data, test_data)
            LinearSvc = LinearSVC().fit(TfidfTrans, train_labels)
            test_predict_LinearSvc_tf = LinearSvc.predict(TfidfTrans_test)
            score = LinearSvc.score(TfidfTrans_test, test_labels)
            matrix = confusion_matrix(test_labels,test_predict_LinearSvc_tf,labels=["YES", "NO"])
            precision, recall, f_measure = fmeasure(matrix)
            return(score, matrix, precision, recall, f_measure)
        elif representation = "count":
            CountTrans, CountTest = countvect(train_data, test_data)
            LinearSvc_count = LinearSVC().fit(CountTrans, train_labels)
            test_predict_LinearSvc_count = LinearSvc_count.predict(CountTest)
            score = LinearSvc_count.score(CountTest, test_labels)
            matrix = confusion_matrix(test_labels,test_predict_LinearSvc_count,labels=["YES", "NO"])
            precision, recall, f_measure = fmeasure(matrix)
            return(score, matrix, precision, recall, f_measure)
        elif representation = "hash":
            HashTrans, HashTest = hashing(train_data, test_data)
            LinearSvc_hash = LinearSVC().fit(HashTrans, train_labels)
            test_predict_LinearSvc_hash = LinearSvc_hash.predict(HashTest)
            score = LinearSvc_hash.score(HashTest, test_labels)
            matrix = confusion_matrix(test_labels,test_predict_LinearSvc_hash,labels=["YES", "NO"])
            precision, recall, f_measure = fmeasure(matrix)
            return(score, matrix, precision, recall, f_measure)
        else:
            sys.exit("This representation has not been implemented yet.")
    elif extra == "nusvc":
        if representation == "tfidf":
            TfidfTrans, TfidfTrans_test = tfidf(train_data, test_data)
            NuSvc_tf = NuSVC().fit(TfidfTrans, train_labels)
            test_predict_NuSVC_tf = NuSvc_tf.predict(TfidfTrans_test)
            score = NuSvc_tf.score(TfidfTrans_test, test_labels)
            matrix = confusion_matrix(test_labels,test_predict_NuSVC_tf,labels=["YES", "NO"])
            precision, recall, f_measure = fmeasure(matrix)
            return(score, matrix, precision, recall, f_measure)
        elif representation == "count":
            CountTrans, CountTest = countvect(train_data, test_data)
            NuSvc_count = NuSVC().fit(CountTrans, train_labels)
            test_predict_NuSVC_count = NuSvc_count.predict(CountTest)
            score = NuSvc_count.score(CountTest, test_labels)
            matrix = confusion_matrix(test_labels,test_predict_NuSVC_count,labels=["YES", "NO"])
            precision, recall, f_measure = fmeasure(matrix)
            return(score, matrix, precision, recall, f_measure)
        elif representation == "hash":
            HashTrans, HashTest = hashing(train_data, test_data)
            NuSvc_hash = NuSVC().fit(HashTrans, train_labels)
            test_predict_NuSVC_hash = NuSvc_hash.predict(HashTest)
            score = NuSvc_hash.score(HashTest, test_labels)
            matrix = confusion_matrix(test_labels,test_predict_NuSVC_hash,labels=["YES", "NO"])
            precision, recall, f_measure = fmeasure(matrix)
            return(score, matrix, precision, recall, f_measure)
        else:
            sys.exit("This representation has not been implemented yet.")
    else:
        sys.exit("This classifier has not been implemented yet.")
        
# function which classifies data using MLP Neural Net classifier      
def neural(train_data, train_labels, test_data, test_labels, representation):
    if representation == "tfidf":
        TfidfTrans, TfidfTrans_test = tfidf(train_data, test_data)
        MLP_tf = MLPClassifier().fit(TfidfTrans, train_labels)
        test_predict_MLP_tf = MLP_tf.predict(TfidfTrans_test)
        score = MLP_tf.score(TfidfTrans_test, test_labels)
        matrix = confusion_matrix(test_labels,test_predict_MLP_tf,labels=["YES", "NO"])
        precision, recall, f_measure = fmeasure(matrix)
        return(score, matrix, precision, recall, f_measure)
    elif representation == "count":
        CountTrans, CountTest = countvect(train_data, test_data)
        MLP_count = MLPClassifier().fit(CountTrans, train_labels)
        test_predict_MLP_count = MLP_count.predict(CountTest)
        score = MLP_count.score(CountTest, test_labels)
        matrix = confusion_matrix(test_labels,test_predict_MLP_count,labels=["YES", "NO"])
        precision, recall, f_measure = fmeasure(matrix)
        return(score, matrix, precision, recall, f_measure)
    elif representation == "hash":
        HashTrans, HashTest = hashing(train_data, test_data)
        MLP_hash = MLPClassifier().fit(HashTrans, train_labels)
        test_predict_MLP_hash = MLP_hash.predict(HashTest)
        score = MLP_hash.score(HashTest, test_labels)
        matrix = confusion_matrix(test_labels,test_predict_MLP_hash,labels=["YES", "NO"])
        precision, recall, f_measure = fmeasure(matrix)
        return(score, matrix, precision, recall, f_measure)
    # if another representation is given as a parameter
    else:
        sys.exit("This classifier has not been implemented yet.")

# function which classifies data using the Scikit version of Naive Bayes
def naive(train_data, train_labels, test_data, test_labels, representation):
    if representation == "tfidf":
        TfidfTrans, TfidfTrans_test = tfidf(train_data, test_data)
        naive_tf = MultinomialNB().fit(TfidfTrans, train_labels)
        test_predict_naive_tf = naive_tf.predict(TfidfTrans_test)
        score = naive_tf.score(TfidfTrans_test, test_labels)
        matrix = confusion_matrix(test_labels, test_predict_naive_tf, labels=["YES", "NO"])
        precision, recall, f_measure = fmeasure(matrix)
        return(score, matrix, precision, recall, f_measure)
    elif representation == "count":
        CountTrans, CountTest = countvect(train_data, test_data)
        naive_count = MultinomialNB().fit(CountTrans, train_labels)
        test_predict_naive_count = naive_count.predict(CountTest)
        score = naive_count.score(CountTest, test_labels)
        matrix = confusion_matrix(test_labels, test_predict_naive_count, labels=["YES", "NO"])
        precision, recall, f_measure = fmeasure(matrix
        return(score, matrix, precision, recall, f_measure)
    elif representation == "hash":
        HashTrans, HashTest = hashing(train_data, test_data)
        naive_hash = MultinomialNB().fit(HashTrans, train_labels)
        test_predict_naive_hash = naive_hash.predict(HashTest)
        score = naive_hash.score(HashTest, test_labels))
        matrix = confusion_matrix(test_labels,test_predict_naive_hash,labels=["YES", "NO"])
        precision, recall, f_measure = fmeasure(matrix)
        return(score, matrix, precision, recall, f_measure)
    # if another representation is given as a parameter
    else:
        sys.exit("This classifier has not been implemented yet.")

# function which classifies data using the Scikit version of Maximum Entropy
def max_ent(train_data, train_labels, test_data, test_labels, representation):
    if representation == "tfidf":
        TfidfTrans, TfidfTrans_test = tfidf(train_data, test_data)
        logit_tf = LogisticRegression().fit(TfidfTrans, train_labels)
        test_predict_tf = logit_tf.predict(TfidfTrans_test)
        score = logit_tf.score(TfidfTrans_test, test_labels)
        matrix = confusion_matrix(test_labels, test_predict_tf, labels=["YES", "NO"])
        precision, recall, f_measure = fmeasure(matrix)
        return(score, matrix, precision, recall, f_measure)
    elif representation == "count":
        CountTrans, CountTest = countvect(train_data, test_data)
        logit_count = LogisticRegression().fit(CountTrans, train_labels)
        test_predict_count = logit_count.predict(CountTest)
        score = logit_count.score(CountTest, test_labels)
        matrix = confusion_matrix(test_labels, test_predict_count, labels=["YES", "NO"])
        precision, recall, f_measure = fmeasure(matrix)
        return(score, matrix, precision, recall, f_measure)
    elif representation == "hash":
        HashTrans, HashTest = hashing(train_data, test_data)
        logit_hash = LogisticRegression().fit(HashTrans, train_data)
        test_predict_hash = logit_hash.predict(HashTest)
        score = logit_hash.score(HashTest, test_labels)
        matrix = confusion_matrix(test_labels,test_predict_hash,labels=["YES", "NO"])
        precision, recall, f_measure = fmeasure(matrix)
        return(score, matrix, precision, recall, f_measure)
    else:
        sys.exit("This classifier has not been implemented yet.")

        
