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

# Implementation of the classifier to detect analogy.
# From http://www.nltk.org/book/ch06.html

analogy_file_name = "test_extractions/analogy_samples.txt"
non_analogy_file_name = "test_extractions/nonanalogy_samples.txt"

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

analogy_list = get_list_re(analogy_file_name)
non_analogy_list = get_list_re(non_analogy_file_name)

#print(get_list_re("test_extractions/prove.txt"))
#print(analogy_list[157])

# labeled data.
samples = [(text, 'YES') for text in analogy_list] + [(text, 'NO') for text in non_analogy_list]
mid_point = int(len(samples) / 2)
random.shuffle(samples)

# verify that the classifier is implemented
mode = input("Please specify the type of classifer you want to use: svm/naive/max_ent/neural? ")
if (mode != "svm") and (mode != "naive") and (mode != "max_ent") and (mode != "neural"):
    sys.exit("This classifier has not been implemented yet.")

# divide data into training set and test set
#feature_sets = [(get_analogy_string(text, mode), label) for (text, label) in samples]
#feature_sets = [(text, label) for (text, label) in samples]
#mid_point = int(len(samples) / 2)
#train_set =  feature_sets[: mid_point]
#test_set = feature_sets[mid_point :]

# change behavior according to the classifier
if mode == "naive":
    # preparing the data
    feature_sets = [({"text:" : text}, label) for (text, label) in samples]
    train_set =  feature_sets[: mid_point]
    test_set = feature_sets[mid_point :]
    # train classifier with training set
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    # test classifier
    print(nltk.classify.accuracy(classifier, test_set))
    print(classifier.show_most_informative_features(1))

elif mode == "svm":
    # Preparing the data
    feature_sets = [(text, label) for (text, label) in samples]
    train_set =  feature_sets[: mid_point]
    test_set = feature_sets[mid_point :]
    train_data = [text for (text, label) in train_set]
    train_labels = [label for (text, label) in train_set]
    test_data = [text for (text, label) in test_set]
    test_labels = [label for (text, label) in test_set]
    # Transforming the data using tf-idf
    TfidfVect = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    TfidfTrans = TfidfVect.fit_transform(train_data)
    TfidfTrans_test = TfidfVect.transform(test_data)
    # Transforming the data using count vectorizer
    CountVect = CountVectorizer(lowercase=False)
    CountTrans = CountVect.fit_transform(train_data)
    CountTest = CountVect.transform(test_data)
    # Transforming the data using hashing vectorizer
    HashVect = HashingVectorizer(lowercase=False)
    HashTrans = HashVect.fit_transform(train_data)
    HashTest = HashVect.transform(test_data)
    # LinearSVC with tf-idf
    LinearSvc = LinearSVC().fit(TfidfTrans, train_labels)
    test_predict_LinearSvc_tf = LinearSvc.predict(TfidfTrans_test)
    # LinearSvc with count vectorizer
    LinearSvc_count = LinearSVC().fit(CountTrans, train_labels)
    test_predict_LinearSvc_count = LinearSvc_count.predict(CountTest)
    # LinearSvc with hash vectorizer
    LinearSvc_hash = LinearSVC().fit(HashTrans, train_labels)
    test_predict_LinearSvc_hash = LinearSvc_hash.predict(HashTest)
    # SVC with td-idf
    Svc_tf = SVC().fit(TfidfTrans, train_labels)
    test_predict_Svc_tf = Svc_tf.predict(TfidfTrans_test)
    # SVC with count vectorizer
    Svc_count = SVC().fit(CountTrans, train_labels)
    test_predict_Svc_count = Svc_count.predict(CountTest)
    # SVC with hash vectorizer
    Svc_hash = SVC().fit(HashTrans, train_labels)
    test_predict_Svc_hash = Svc_hash.predict(HashTest)
    # NuSVC with tf-idf
    NuSvc_tf = NuSVC().fit(TfidfTrans, train_labels)
    test_predict_NuSVC_tf = NuSvc_tf.predict(TfidfTrans_test)
    # NuSVC with count
    NuSvc_count = NuSVC().fit(CountTrans, train_labels)
    test_predict_NuSVC_count = NuSvc_count.predict(CountTest)
    # NuSVC with hash vectorizer
    NuSvc_hash = NuSVC().fit(HashTrans, train_labels)
    test_predict_NuSVC_hash = NuSvc_hash.predict(HashTest)
                              
    # Prints
    print("LinearSvc with tf-idf: ", LinearSvc.score(TfidfTrans_test, test_labels))
    print("Confusion matrix for LinearSVC with tf-idf :\n", confusion_matrix(test_labels,test_predict_LinearSvc_tf,labels=["YES", "NO"]))
    print("LinearSvc with count vec: ", LinearSvc_count.score(CountTest, test_labels))
    print("Confusion matrix for LinearSVC with count :\n", confusion_matrix(test_labels,test_predict_LinearSvc_count,labels=["YES", "NO"]))
    print("LinearSvc with hash vec: ", LinearSvc_hash.score(HashTest, test_labels))
    print("Confusion matrix for LinearSVC with hash :\n", confusion_matrix(test_labels,test_predict_LinearSvc_hash,labels=["YES", "NO"]))
    
    print()
    print("Svc with tf-idf: ", Svc_tf.score(TfidfTrans_test, test_labels))
    print("Confusion matrix for SVC with tf-idf :\n", confusion_matrix(test_labels,test_predict_Svc_tf,labels=["YES", "NO"]))
    print("Svc with count: ", Svc_count.score(CountTest, test_labels))
    print("Confusion matrix for SVC with count :\n", confusion_matrix(test_labels,test_predict_Svc_count,labels=["YES", "NO"]))
    print("Svc with hash: ", Svc_hash.score(HashTest, test_labels))
    print("Confusion matrix for SVC with hash :\n", confusion_matrix(test_labels,test_predict_Svc_hash,labels=["YES", "NO"]))
        
    print()
    print("NuSvc with tf-idf: ", NuSvc_tf.score(TfidfTrans_test, test_labels))
    print("Confusion matrix for NuSVC with tf-idf :\n", confusion_matrix(test_labels,test_predict_NuSVC_tf,labels=["YES", "NO"]))
    print("NuSvc with count: ", NuSvc_count.score(CountTest, test_labels))
    print("Confusion matrix for NuSVC with count :\n", confusion_matrix(test_labels,test_predict_NuSVC_count,labels=["YES", "NO"]))
    print("NuSvc with hash: ", NuSvc_hash.score(HashTest, test_labels))
    print("Confusion matrix for NuSVC with hash :\n", confusion_matrix(test_labels,test_predict_NuSVC_hash,labels=["YES", "NO"]))
               
elif mode == "max_ent":
    # preparing the data
    feature_sets = [({"text:" : text}, label) for (text, label) in samples]
    train_set =  feature_sets[: mid_point]
    test_set = feature_sets[mid_point :]
    max_ent = nltk.classify.MaxentClassifier.train(train_set, 'GIS', trace=0, max_iter=1000)
    print("Maximum Entropy: nltk.classify.accuracy(max_ent, test_set)")
    #print(max_ent.show_most_informative_features(5))

elif mode == "neural":
    # Preparing the data
    feature_sets = [(text, label) for (text, label) in samples]
    train_set =  feature_sets[: mid_point]
    test_set = feature_sets[mid_point :]
    train_data = [text for (text, label) in train_set]
    train_labels = [label for (text, label) in train_set]
    test_data = [text for (text, label) in test_set]
    test_labels = [label for (text, label) in test_set]
    # Transforming the data using tf-idf
    TfidfVect = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    TfidfTrans = TfidfVect.fit_transform(train_data)
    TfidfTrans_test = TfidfVect.transform(test_data)
    # Transforming the data using count vectorizer
    CountVect = CountVectorizer(lowercase=False)
    CountTrans = CountVect.fit_transform(train_data)
    CountTest = CountVect.transform(test_data)
    # Transforming the data using hashing vectorizer
    HashVect = HashingVectorizer(lowercase=False)
    HashTrans = HashVect.fit_transform(train_data)
    HashTest = HashVect.transform(test_data)
    # Neural Networks with tf-idf
    MLP_tf = MLPClassifier().fit(TfidfTrans, train_labels)
    test_predict_MLP_tf = MLP_tf.predict(TfidfTrans_test)
    print("Neural Networks using MLP with tf-idf: ", MLP_tf.score(TfidfTrans_test, test_labels))
    print("Confusion matrix for MLP with tf-idf :\n", confusion_matrix(test_labels,test_predict_MLP_tf,labels=["YES", "NO"]))
    # Neural Networks with count
    MLP_count = MLPClassifier().fit(CountTrans, train_labels)
    test_predict_MLP_count = MLP_count.predict(CountTest)
    print("Neural Networks using MLP with count: ", MLP_count.score(CountTest, test_labels))
    print("Confusion matrix for MLP with count :\n", confusion_matrix(test_labels,test_predict_MLP_count,labels=["YES", "NO"]))
    # Neural Networks with hash
    MLP_hash = MLPClassifier().fit(HashTrans, train_labels)
    test_predict_MLP_hash = MLP_hash.predict(HashTest)
    print("Neural Networks using MLP with hash: ", MLP_hash.score(HashTest, test_labels))
    print("Confusion matrix for MLP with hash :\n", confusion_matrix(test_labels,test_predict_MLP_hash,labels=["YES", "NO"]))


    
    
    
    



