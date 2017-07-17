import nltk
from analogy_strings import analogy_string_list
from sentence_parser import get_speech_tags
from personal import root
#------------------------
from nltk.classify import SklearnClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.neural_network import MLPClassifier
from nltk.classify import maxent
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

def get_list_re(filename):
    # Returns all training data as a list
    # File should be formatted as a text line followed '>' in the next line
    # before a new text line.
    list = []
    file = open(filename, "r", encoding = "utf-8")
    for line in file.readlines():
        if line[0] != '>' and line != "\n":
            l = line.split("]\",")[-1]
            final = re.sub("[^a-zA-Z]"," ", l)
            #final = line.split("]\",")[-1].split(".")[0].split("?")[0].split("!")[0]
            list.append(final)

    return list

# labeled data.
samples = [(text, 'YES') for text in analogy_list] + [(text, 'NO') for text in non_analogy_list]

def preprocess_svm_neural(samples):
    num_samples = len[samples]
    random.shuffle(samples)
    cutoff = int((1.0 - percent_test) * num_samples)
    feature_sets = [(text, label) for (text, label) in samples]
    train_set =  feature_sets[:cutoff]
    test_set = feature_sets[cutoff:]
    train_data = [text for (text, label) in train_set]
    train_labels = [label for (text, label) in train_set]
    test_data = [text for (text, label) in test_set]
    test_labels = [label for (text, label) in test_set]
    return(train_data, train_labels, test_data, test_labels)

def preprocess_naive_max(samples):
    num_samples = len[samples]
    random.shuffle(samples)
    cutoff = int((1.0 - percent_test) * num_samples)
    feature_sets = [({"text:" : text}, label) for (text, label) in samples]
    train_set =  feature_sets[:cutoff]
    test_set = feature_sets[cutoff:]
    return(train_set, test_set)
    
def tfidf(train_data, test_data):
    TfidfVect = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    TfidfTrans = TfidfVect.fit_transform(train_data)
    TfidfTrans_test = TfidfVect.transform(test_data)
    return(TfidfTrans, TfidfTrans_test)

def countvect(train_data, test_data):
    CountVect = CountVectorizer(lowercase=False)
    CountTrans = CountVect.fit_transform(train_data)
    CountTest = CountVect.transform(test_data)
    return(CountTrans, CountTest)

def hashing(train_data, test_data):
    HashVect = HashingVectorizer(lowercase=False)
    HashTrans = HashVect.fit_transform(train_data)
    HashTest = HashVect.transform(test_data)
    return(HashTrans, HashTest)

# within the if classifier = nn statement
train_data, train_labels, test_data, test_labels = preprocess_svm_neural(samples)
def svm(train_data, train_labels, test_data, test_labels, representation, extra=[]):
    # if the classifier not specified, use SVC as the default one
    if extra == "":
        if representation == "tfidf":
            TfidfTrans, TfidfTrans_test = tfidf(train_data, test_data)
            Svc_tf = SVC().fit(TfidfTrans, train_labels)
            test_predict_Svc_tf = Svc_tf.predict(TfidfTrans_test)
            score = Svc_tf.score(TfidfTrans_test, test_labels)
            matrix = confusion_matrix(test_labels,test_predict_Svc_tf,labels=["YES", "NO"])
            return(score, matrix)
        elif representation == "count":
            CountTrans, CountTest = countvect(train_data, test_data)
            Svc_count = SVC().fit(CountTrans, train_labels)
            test_predict_Svc_count = Svc_count.predict(CountTest)
            score = Svc_tf.score(TfidfTrans_test, test_labels)
            matrix = confusion_matrix(test_labels,test_predict_Svc_hash,labels=["YES", "NO"])
            return(score, matrix)
        elif representation == "hash":
            HashTrans, HashTest = hashing(train_data, test_data)
            Svc_hash = SVC().fit(HashTrans, train_labels)
            test_predict_Svc_hash = Svc_hash.predict(HashTest)
            score = Svc_hash.score(HashTest, test_labels)
            matrix = confusion_matrix(test_labels,test_predict_Svc_hash,labels=["YES", "NO"])
            return(score, matrix)
        else:
            sys.exit("This representation has not been implemented yet.")
    # if the classifier is specified
    elif extra == "linear":
        if representation == "tfidf":
            TfidfTrans, TfidfTrans_test = tfidf(train_data, test_data)
            LinearSvc = LinearSVC().fit(TfidfTrans, train_labels)
            test_predict_LinearSvc_tf = LinearSvc.predict(TfidfTrans_test)
            score = LinearSvc.score(TfidfTrans_test, test_labels)
            matrix = confusion_matrix(test_labels,test_predict_LinearSvc_tf,labels=["YES", "NO"])
            return(score, matrix)
        elif representation = "count":
            CountTrans, CountTest = countvect(train_data, test_data)
            LinearSvc_count = LinearSVC().fit(CountTrans, train_labels)
            test_predict_LinearSvc_count = LinearSvc_count.predict(CountTest)
            score = LinearSvc_count.score(CountTest, test_labels)
            matrix = confusion_matrix(test_labels,test_predict_LinearSvc_count,labels=["YES", "NO"])
            return(score, matrix)
        elif representation = "hash":
            HashTrans, HashTest = hashing(train_data, test_data)
            LinearSvc_hash = LinearSVC().fit(HashTrans, train_labels)
            test_predict_LinearSvc_hash = LinearSvc_hash.predict(HashTest)
            score = LinearSvc_hash.score(HashTest, test_labels)
            matrix = confusion_matrix(test_labels,test_predict_LinearSvc_hash,labels=["YES", "NO"])
            return(score, matrix)
        else:
            sys.exit("This representation has not been implemented yet.")
    elif extra == "nusvc":
        if representation == "tfidf":
            TfidfTrans, TfidfTrans_test = tfidf(train_data, test_data)
            NuSvc_tf = NuSVC().fit(TfidfTrans, train_labels)
            test_predict_NuSVC_tf = NuSvc_tf.predict(TfidfTrans_test)
            score = NuSvc_tf.score(TfidfTrans_test, test_labels)
            matrix = confusion_matrix(test_labels,test_predict_NuSVC_tf,labels=["YES", "NO"])
            return(score, matrix)
        elif representation == "count":
            CountTrans, CountTest = countvect(train_data, test_data)
            NuSvc_count = NuSVC().fit(CountTrans, train_labels)
            test_predict_NuSVC_count = NuSvc_count.predict(CountTest)
            score = NuSvc_count.score(CountTest, test_labels)
            matrix = confusion_matrix(test_labels,test_predict_NuSVC_count,labels=["YES", "NO"])
            return(score, matrix)
        elif representation == "hash":
            HashTrans, HashTest = hashing(train_data, test_data)
            NuSvc_hash = NuSVC().fit(HashTrans, train_labels)
            test_predict_NuSVC_hash = NuSvc_hash.predict(HashTest)
            score = NuSvc_hash.score(HashTest, test_labels)
            matrix = confusion_matrix(test_labels,test_predict_NuSVC_hash,labels=["YES", "NO"])
            return(score, matrix)
        else:
            sys.exit("This representation has not been implemented yet.")
    else:
        sys.exit("This classifier has not been implemented yet.")
        
# within the other if statement in main_interface        
train_data, train_labels, test_data, test_labels = preprocess_svm_neural(samples)        
def neural(train_data, train_labels, test_data, test_labels, representation):
    if representation = "tfidf":
        MLP_tf = MLPClassifier().fit(TfidfTrans, train_labels)
        test_predict_MLP_tf = MLP_tf.predict(TfidfTrans_test)
        score = MLP_tf.score(TfidfTrans_test, test_labels)
        matrix = confusion_matrix(test_labels,test_predict_MLP_tf,labels=["YES", "NO"])
        return(score, matrix)
    elif representation = "count":
        MLP_count = MLPClassifier().fit(CountTrans, train_labels)
        test_predict_MLP_count = MLP_count.predict(CountTest)
        score = MLP_count.score(CountTest, test_labels)
        matrix = confusion_matrix(test_labels,test_predict_MLP_count,labels=["YES", "NO"])
        return(score, matrix)
    elif representation = "hash":
        MLP_hash = MLPClassifier().fit(HashTrans, train_labels)
        test_predict_MLP_hash = MLP_hash.predict(HashTest)
        score = MLP_hash.score(HashTest, test_labels))
        matrix confusion_matrix(test_labels,test_predict_MLP_hash,labels=["YES", "NO"]))
        return(score, matrix)
    else:
        sys.exit("This classifier has not been implemented yet.")

# within the other if statement 
train_set, test_set = preprocess_naive_max(samples)
def naive(train_set, test_set):
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    score = nltk.classify.accuracy(classifier, test_set)
    return(score)

# within the other if statement
train_set, test_set = preprocess_naive_max(samples)
def max_ent(train_set, test_set):
    max_ent = nltk.classify.MaxentClassifier.train(train_set, 'GIS', trace=0, max_iter=1000)
    score = nltk.classify.accuracy(max_ent, test_set)
    return(score)