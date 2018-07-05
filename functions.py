# import nltk
# from analogy_strings import analogy_string_list
# from sentence_parser import get_speech_tags
# from personal import root
#------------------------
from nltk.classify import SklearnClassifier
from sklearn.feature_extraction import DictVectorizer
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
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import json
import operator
from timeout import timeout
import numpy as np
import pandas as pd
#------------------------
import random
import re
import csv
from nltk.corpus import wordnet as wn
from boyer_moore import find_boyer_moore
#-------------------------
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif


modifier = ['amod','nmod']
"""To add new feature, add new function that takes a train and a test set and extra and return a train and a test set based on what representation you want ot use"""
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
    sent = []
    with open(filename) as file:
        readcsv = csv.reader(file, delimiter=',')
        for row in readcsv:
            sentence = row[1]
            sent.append(sentence)
    return sent

def read_CSV(csvfile, r):
    lst = []
    with open(csvfile) as file:
        readcsv = csv.reader(file, delimiter=',')
        for row in readcsv:
            sentence = row[r]
            lst.append(sentence)
    return lst



def explore_csv(bt_parsed, bt_label, tree_type):
    sentences = bt_parsed
    d_type = {}
    for i in range(6):
        d_type[str(i)] = 0
    data = defaultdict(lambda :0)
    for i in range(756):
        data[str(i)] = {"correct_predicted": 0, "false_predicted": 0, "test_appearance": 0}
    lst_type = []
    pos_sent = read_CSV('base_target_pos.csv',1)
    neg_sent = read_CSV('base_target_neg.csv',1)
    count = 0
    for seed in range(100):
        d = {}
        c = 0
        for i in range(6):
            d[str(i)] = 0
        sent_csv = read_CSV('./testing/prediction' + str(seed)+ '.csv',1)
        false_csv = read_CSV('./testing/false' + str(seed) + '.csv',1)
        test_sent = read_CSV('./testing/test_set' + str(seed)+ '.csv',1)
        sent_csv = sent_csv[1:]
        false_csv = false_csv[1:]
        test_sent = test_sent[1:]
        if test_sent == false_csv + sent_csv:
            count += 1
        for sent, label, tp in zip(sentences, bt_label, tree_type):
            temp_true = int(data[str(c)]["correct_predicted"])
            temp_appr = int(data[str(c)]["test_appearance"])
            if str(sent) in sent_csv:
                if str(sent) in pos_sent:
                    temp_true += 1
                    d[str(tp)] += 1
            if str(sent) in test_sent:
                temp_appr += 1
                tpe = sent["tree_type"]
                d_type[tpe] += 1
            if str(sent) in false_csv:
                if str(sent) in neg_sent:
                    temp_true += 1
                    d[str(tp)] += 1
            dic= {'data': sent, 'label': label["label"], 'tree_type': tp, 'correct_predicted': temp_true,'test_appearance': temp_appr}
            data[str(c)] = dic
            c += 1
        # print(d)
        lst_type.append(d)
    d = {}
    for i in range(6):
        d[str(i)] = 0
    for d_i in lst_type:
        for i in d_i:
            d[i] += d_i[i]
    for i in d:
        d[i] = d[i]/d_type[i]
    print(d)
    writeJSON(data, './testing/data.json')

def explore_tree_type():
    d = {}
    for i in range(6):
        d[str(i)] = {"YES": 0, "NO":0}
    tree_type = read_CSV('base_target.csv',4)
    label = read_CSV('base_target.csv',5)
    for tpe, lab in zip(tree_type,label):
        d[tpe][lab] += 1
    for key in d:
        d[key]["YES"] = d[key]["YES"]/ (d[key]["YES"] + d[key]["NO"])
        d[key]["NO"] = 1 - d[key]["YES"]
    print(d)
def explore_parser():
    tree_parse_count, dep_parse_count, count = 0,0,0
    for seed in range(100):
        pos_count,neg_count = -1,-1
        false_csv = read_CSV('./testing/false' + str(seed) + '.csv', 1)
        true_csv = read_CSV('./testing/prediction' + str(seed)+ '.csv',1)
        label_false_csv = read_CSV('./testing/false' + str(seed) + '.csv',3)
        label_true_csv =  read_CSV('./testing/prediction' + str(seed) + '.csv',3)
        for dat_pos, dat_neg, lab_pos, lab_neg in zip(true_csv, false_csv, label_true_csv, label_false_csv):
            if pos_count >=0 and neg_count >=0:
                if "'tree_detected': '1'" in dat_pos:
                    if lab_pos == 'YES':
                        tree_parse_count += 1
                if  "'tree_detected': '0'" in dat_neg:
                    if lab_neg == 'NO':
                        tree_parse_count += 1
                if "'detected': True" in dat_pos:
                    if lab_pos == 'YES':
                        dep_parse_count += 1
                if  "'detected': False" in dat_neg:
                    if lab_neg == 'NO':
                        dep_parse_count += 1
                count += 1
            count+= 1
            pos_count += 1
            neg_count += 1
    print(count)
    print("tree parse: ",tree_parse_count,"___", tree_parse_count/count)
    print("dep parse", dep_parse_count,"____", dep_parse_count/count)

# preprocess the data so it can be used by the classifiers
def preprocess(samples, percent_test,seed, num_samples, caller= ''):
    if caller == 'test_main_interface_output': 
        random.seed(seed)
    random.shuffle(samples)
    cutoff = int((1.0 - percent_test) * num_samples)
    train_data, test_data, train_labels, test_labels = [], [], [], []
    y_count, n_count = 0,0
    train_set, test_set = [],[]
    for key in samples:
        if y_count > cutoff/2 :
            break
        elif key["label"] == 'YES':
            train_set.append(key)
            y_count += 1
    for key in samples:
        if n_count > cutoff/2 :
            break
        elif key["label"] == 'NO':
            train_set.append(key)
            n_count += 1
    for key in samples:
        if key not in train_set:
            test_set.append(key)
    train_data,train_labels = divide_data_labels(train_data,train_labels,train_set)
    test_data, test_labels = divide_data_labels(test_data, test_labels, test_set)
    # separate the training data and the training labels
    return(train_data, train_labels, test_data, test_labels)

def divide_data_labels(data,label, set):
    for key in set:
        dic = {}
        for k in key:
            if k == "label":
                label.append(key["label"])
            else:
                dic[k] = key[k]
        data.append(dic)
    return data,label

# Transform the data so it can be represented by prepositional phrases.
def preposition(train_data, test_data):
    pp_training_set = [get_pp(text) for text in train_data]
    pp_test_set = [get_pp(text) for text in test_data]
    dict_vect = DictVectorizer()
    PpTrans = dict_vect.fit_transform(pp_training_set)
    PpTest = dict_vect.transform(pp_test_set)
    return (PpTrans, PpTest)

def base_target_pair(train_data, test_data, extra, train_labels, test_labels):
    """refer to DependencyParsing.py to change/add features to the model"""
    dict_vect = DictVectorizer()
    bt_train = dict_vect.fit_transform(train_data)
    bt_test = dict_vect.transform(test_data)
    # print(bt_training, bt_testing)
    return (bt_train, bt_test)
    # return training_txt

def delete_element(count,o_list):
    lst = []
    for i in range(len(o_list)):
        if isFalse(i,count):
            lst.append(o_list[i])
    return lst

def isFalse(i,s):
    for j in s:
        if i == j:
            return True
    return False

def writeJSON(dic, dire):
    with open(dire, 'w') as fp:
        json.dump(dic, fp, indent= 4)

def writeCSVFile(text_output, to_dir):
    f = open(to_dir, 'w')
    for line in text_output:
            f.write(line)
    f.close()

def readCSV(csvFile, method, csvTree = './base_target_tree.csv'):
    lst = []
    with open(csvFile) as file, open(csvTree) as tree:
        readcsv = csv.reader(file, delimiter=',')
        readTree = csv.reader(tree, delimiter = ",")
        for row,row1 in zip(readcsv,readTree):
            # ide = row[0]
            sentence = row[2]
            b = row[0]
            t = row[1]
            similarity  = row[3]
            tree_type = row[4]
            label = row[5]
            tree_detected = row1[1]
            detected = True
            if len(t) == 0 and len(b) == 0:
                detected = False
            if method == 1:
                lst.append({"sentence": sentence,   "tree_detected": tree_detected, "label": label})
            elif method == 0:
                lst.append({"sentence": sentence,  "tree_detected": tree_detected})
            elif method == 2:
                lst.append({"label": label})
            elif method == 3:
                lst.append({"sentence": sentence, "base": b, "target": t, "similarity": similarity, "tree_type": tree_type, "label": label})
            else:
                lst.append({"sentence": sentence, "label": label})
    return lst

def wordForm(word):
    w = None
    try:
        w = wn.synset(word + '.n.01')
    except:
        print(word)
    return w

def getIndex(word,sentence):
    lst = sentence.split()
    ind = 0
    for w in lst:
        if w.lower() == word.lower():
            return ind
        ind += 1
# Transform the data so it can be represented using tfidf
def tfidf(train_data, test_data, extra):
    TfidfVect = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False, stop_words=extra['stop_words'], max_df=extra['max_df'], norm=extra['norm'], min_df=extra['min_df'])
    TfidfTrans = TfidfVect.fit_transform(train_data)
    TfidfTrans_test = TfidfVect.transform(test_data)
    return(TfidfTrans, TfidfTrans_test)

# Transform the data so it can be represented using Count Vectorizer
def countvect(train_data, test_data, extra):
    CountVect = CountVectorizer(lowercase=False, stop_words=extra['stop_words'], max_df=extra['max_df'])
    CountTrans = CountVect.fit_transform(train_data)
    CountTest = CountVect.transform(test_data)
    return(CountTrans, CountTest)

# Transform the data so it can be represented using Hashing Vectorizer
def hashing(train_data, test_data,extra, classifier=[]):
    if classifier == "naive":
        HashVect = HashingVectorizer(lowercase=False, non_negative=True, stop_words=extra['stop_words'],norm=extra['norm'])
    else:
         HashVect = HashingVectorizer(lowercase=False, stop_words=extra['stop_words'], norm=extra['norm'])
    HashTrans = HashVect.fit_transform(train_data)
    HashTest = HashVect.transform(test_data)
    return(HashTrans, HashTest)

# Implementetion of the fmeasure metric, which calculates the precision, recall and f1measure given a confusion matrix
def fmeasure(matrix):
    """f1 score"""
    value1 = (matrix[0][1] + matrix[0][0])
    value2 = (matrix[1][0] + matrix[0][0])
    if value1 == 0 or value2 == 0:
        precision = 0
        recall = 0
        f_measure = 0
    else:
        precision = matrix[0][0] / value2
        recall = matrix[0][0] / value1
        if precision == 0:
            f_measure = 0
        else:
            f_measure = (2 * precision * recall) / (precision + recall)
    return(precision, recall, f_measure)

def classify(train_data, train_labels, test_data, test_labels, classifier_name, representation, extra={"sub_class":""}, time=1000000000):
    @timeout(time)
    def _classify(train_data, train_labels, test_data, test_labels, classifier_name, representation, extra):
        clfier = get_classifier(classifier_name, extra)
        train_set, test_set = get_representation(train_data, test_data, representation, classifier_name, extra)
        learn_results = clfier.fit(train_set, train_labels)
        score = learn_results.score(test_set, test_labels)
        test_predict = learn_results.predict(test_set)
        matrix = confusion_matrix(test_labels, test_predict, labels = ['YES', 'NO'])
        precision, recall, f_measure = fmeasure(matrix)
        return (score, matrix, precision, recall, f_measure)
    return _classify(train_data, train_labels, test_data, test_labels, classifier_name, representation, extra)


def get_classifier(name, extra):
    if name == "svm":
        if extra["sub_class"] == "" or extra["sub_class"] == "svc":
            return SVC(kernel=extra['kernel'], max_iter=extra['max_iter_svc'], )
        elif extra["sub_class"] == "linear":
            return LinearSVC(C=extra['C'])
        elif extra["sub_class"] == "nusvc":
            return NuSVC(kernel=extra['kernel'], max_iter=extra['max_iter_svc'], decision_function_shape=extra['decision_function_shape'], degree=extra['degree'], nu=extra['nu'], tol=extra['tol'])
    elif name == "neural":
        return MLPClassifier(hidden_layer_sizes=extra['hidden_layer_sizes'], activation=extra['activation'], solver=extra['solver'], max_iter=extra['max_iter'], early_stopping=extra['early_stopping'], learning_rate=extra['learning_rate'])
    elif name == "naive":
        return MultinomialNB(alpha=extra['alpha'])
    elif name == "max_ent":
        return LogisticRegression(C=extra['C'], max_iter=extra['max_iter_log'], solver=extra['solver_log'])
    else:
        sys.exit("This classifier has not been implemented yet.")
        return None

def get_representation(train_data, test_data, representation, classifier, extra, train_labels, test_labels):
    """representation of data"""
    if representation == "tfidf":
        return tfidf(train_data, test_data, extra)
    elif representation == "count":
        return countvect(train_data, test_data, extra)
    elif representation == "base_target":
        return base_target_pair(train_data,test_data,extra, train_labels, test_labels)
    elif representation == "hash":
        if classifier == "naive":
            return hashing(train_data, test_data, extra, "naive")
        else:
            return hashing(train_data, test_data, extra)
    else:
        sys.exit("This representation has not been implemented yet.")
        return None

def set_default(extra, key, value):
    try:
        if extra[key] == "":
            extra[key] = value
    except KeyError:
        extra[key] = value

def set_extra(extra):
    """set extra parameters"""
    set_default(extra,'sub_class', "")
    set_default(extra,'stop_words', None)
    set_default(extra,'hidden_layer_sizes', 100)
    set_default(extra,'activation', 'relu')
    set_default(extra,'max_df', 1.0) #1.0
    set_default(extra,'min_df', 0.5) #1.0
    set_default(extra,'norm', 'l2')
    set_default(extra,'alpha', 1.0)
    set_default(extra,'kernel', 'rbf')
    set_default(extra,'max_iter', 200)
    set_default(extra,'max_iter_log', 100)
    set_default(extra,'max_iter_linear', 1000)
    set_default(extra,'max_iter_svc', -1)
    set_default(extra,'solver', 'adam')
    set_default(extra,'solver_log', 'liblinear')
    set_default(extra,'early_stopping', False)
    set_default(extra,'C', 1.0)
    set_default(extra,'degree', 3)
    set_default(extra,'learning_rate','constant')
    set_default(extra,'decision_function_shape','ovr')
    set_default(extra,'nu', 0.5)
    set_default(extra,'tol', 0.001)
    set_default(extra, 'word_hypernyms', 0.001)
    return(extra)
def strip_id(lst):
    output = []
    for dic in lst:
        d = {}
        for key in dic:
            if key != "sentence":
                d[key] = dic[key]
        output.append(d)
    return output

def divide_pos_neg():
    bt_parsed = readCSV('base_target.csv',0)
    label = readCSV('base_target.csv',2)
    d1,d2 = {"data": []},{"data": []}
    c,n,p = 0,0,0
    for lab,sent in zip(label,bt_parsed):
        if lab["label"] == 'YES':
            d1["data"].append(sent)
            p  += 1
        elif lab["label"] == "NO":
            d2["data"].append(sent)
            n += 1
        c += 1
    pd.DataFrame(d1, columns=['data']).to_csv('base_target_pos' + '.csv')
    pd.DataFrame(d2, columns=['data']).to_csv('base_target_neg' + '.csv')


def classify_pipeline(train_data, train_labels, test_data, test_labels, classifier_name, representation, seed, extra={"sub_class":""}, time=1000000000):
    @timeout(time)

    def _classify(train_data, train_labels, test_data, test_labels, classifier_name, representation, seed, extra):
        clfier = get_classifier(classifier_name, extra)
        train_set, test_set = get_representation(train_data, test_data, representation, classifier_name, extra, train_labels, test_labels)
        selector = SelectPercentile(f_classif, percentile=100)
        estimators = [('reduce_dim', selector), ('clf', clfier)]
        pipe = Pipeline(estimators)
        learn_results = pipe.fit(train_set, train_labels)

        score = learn_results.score(test_set, test_labels)
        test_predict = learn_results.predict(test_set)
        d = {'sentence': [],'data': [], 'label': [], 'answer': []}
        d2 =  {'data': [], 'label': [], 'answer': []}
        # d = {'data': test_data, 'label': test_predict, 'answer': test_labels}
        for pred, lab, dat in zip(test_predict, test_labels, test_data):
            if pred ==  'YES':
                # d['sentence'].append(dat['sentence'])
                d['data'].append(dat)
                d['label'].append(pred)
                d['answer'].append(lab)
            if pred == 'NO':
                d2['data'].append(dat)
                d2['label'].append(pred)
                d2['answer'].append(lab)
        pd.DataFrame(d, columns=['data','label','answer']).to_csv('./testing/prediction' + str(int(seed)) + '.csv')
        pd.DataFrame(d2, columns= ['data', 'label', 'answer']).to_csv('./testing/false'+ str(int(seed)) + '.csv')
        matrix = confusion_matrix(test_labels, test_predict, labels = ['YES', 'NO'])
        precision, recall, f_measure = fmeasure(matrix)
        return (score, matrix, precision, recall, f_measure)
    return _classify(train_data, train_labels, test_data, test_labels, classifier_name, representation, seed, extra)
