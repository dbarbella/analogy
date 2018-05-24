from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse import stanford
from parser import readFile
path_to_jar = './stanford-parser/jars/stanford-parser.jar'
path_to_models_jar = './stanford-parser/jars/stanford-english-corenlp-2018-02-27-models.jar'
from time import time
import re
parser = stanford.StanfordParser(path_to_models_jar,path_to_jar,encoding = 'utf8')
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
verb = ['VB', 'VBZ', 'VBC' , 'VBN', 'VBP', 'root', 'VBG', 'VBD']
subject = ['dobj', 'nsubj']
noun = ['NP','NN', 'NNP', 'NNS', 'PRP']
tobe = ["was being", "were being" "will be", "is going to", "am going to", "are going to", "has been", "have been", "am", "are", "is", "was", "were"]
def dependency_parse(sentence):
    for v in tobe:
        sentence = sentence.replace('\b'+v+'\b','behave')
    result = dependency_parser.raw_parse(sentence)
    # dep = result.__next__()
    target = None
    base = None
    for line in result:
        target, tar_index = target_search(line.nodes)
        if tar_index is not None:
            base = base_search(tar_index,line.nodes)
    return target,base
    # print(list(dep.triples()))

def target_search(p):
    for i in range(len(p)):
        if p[i]["word"] == 'like':
            index = p[i]["head"]
            return p[index]["word"], index
    return None, None

def base_search(tar_index,line):
    base_index = tar_index
    grand_base_index = line[base_index]["head"]
    if line[grand_base_index]["tag"] in noun and line[grand_base_index]["rel"] == 'dobj':
        for i in range(grand_base_index,tar_index):
            if "compound" in line[i]["deps"] and line[i]["head"] == grand_base_index:
                return line[grand_base_index]["word"]
    while line[base_index]["tag"] not in verb:
        temp = line[base_index]["head"]
        if temp == 0:
            break
        else:
            base_index = temp
    grand_base_index = line[base_index]["head"]
    while line[grand_base_index]["tag"] in verb:
        for i in range(base_index):
            if line[i]["head"] == base_index and line[i]["tag"] in noun and i != tar_index and line[i]["rel"] != "nmod:tmod":
                return line[i]["word"]
        for i in range(len(line)):
            if line[i]["head"] == grand_base_index and line[i]["tag"] in noun and i != tar_index and line[i]["rel"] != "nmod:tmod":
                return line[i]["word"]
        base_index = line[base_index]["head"]
        grand_base_index = line[base_index]["head"]

    if search_WP(line,base_index):
        return line[grand_base_index]["word"]
    else:
        for i in range(len(line)):
            if line[i]["head"] == base_index and line[i]["tag"] in noun and i != tar_index and line[i]["rel"] != "nmod:tmod":
                return line[i]["word"]

def check_numerical(line,index):
    if line[index]["tag"] == 'CD':
        for i in range(index, len(line)):
            if line[i]["head"] == index and line[i]["tag"] in noun:
                return line[i]["word"]


def search_WP(line,head):
    for i in range(len(line)):
        if line[i]["head"] == head:
            if line[i]["tag"] == 'WP':
                return True
    return False



if __name__ == '__main__':
    sentences = readFile('./verified_analogies.csv')
    lower_tie = 0
    upper_tie = 150
    start = time()
    count = 0
    base = []
    target = []
    for i in range(lower_tie,upper_tie):
        print('____', i, '_____')
        print(sentences[i])
        t,b = dependency_parse(sentences[i])
        if t and b is not None:
            count+=1
        base.append(b)
        target.append(t)
        print(b, '________', t)
    print('running time:', time() - start)
    print('detect:', count)

