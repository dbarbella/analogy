from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse import stanford
from parser import readFile, writeTSVFile

path_to_jar = './stanford-parser/jars/stanford-parser.jar'
path_to_models_jar = './stanford-parser/jars/stanford-english-corenlp-2018-02-27-models.jar'
from time import time
import re
parser = stanford.StanfordParser(path_to_models_jar,path_to_jar,encoding = 'utf8')
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
verb = ['VB', 'VBZ', 'VBC' , 'VBN', 'VBP', 'root', 'VBG', 'VBD']
subject = ['dobj', 'nsubj']
noun = ['NP','NN', 'NNP', 'NNS', 'PRP']
linking_words = ["like"]
number = ["tens", "hundreds", "thousands", "millions", "billions", "trillions","dose","dozen"]
tobe = ["was being", "were being" "will be", "is going to", "am going to", "are going to", "has been", "have been", "am", "are", "is", "was", "were"]
def dependency_parse(sentence,like_count):
    for v in tobe:
        sentence = sentence.replace('\b'+v+'\b','behave')
    result = dependency_parser.raw_parse(sentence)
    # dep = result.__next__()
    target = None
    base = None
    for line in result:
        target, tar_index,like_count = target_search(line.nodes,like_count)
        if tar_index is not None:
            base = base_search(tar_index,line.nodes)
    return target,base,like_count
    # print(list(dep.triples()))

def target_search(p,like_count):
    for i in range(len(p)):
        if p[i]["word"] in linking_words:
            index = p[i]["head"]
            like_count += 1
            return check_numerical(p,index), index, like_count
    return None, None, like_count

def base_search(tar_index,line):
    base_index = tar_index
    if base_index == 0:
        return None
    grand_base_index = line[base_index]["head"]
    if line[grand_base_index]["tag"] in noun and line[grand_base_index]["rel"] == 'dobj':
        for i in range(grand_base_index,tar_index):
            if "compound" in line[i]["deps"] and line[i]["head"] == grand_base_index:
                return check_numerical(line,grand_base_index)
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
                return check_numerical(line,i)
        for i in range(len(line)):
            if line[i]["head"] == grand_base_index and line[i]["tag"] in noun and i != tar_index and line[i]["rel"] != "nmod:tmod":
                return check_numerical(line,i)
        base_index = line[base_index]["head"]
        grand_base_index = line[base_index]["head"]

    if search_WP(line,base_index):
        return check_numerical(line,grand_base_index)
    else:
        for i in range(len(line)):
            if line[i]["head"] == base_index and line[i]["tag"] in noun and i != tar_index and line[i]["rel"] != "nmod:tmod":
                return check_numerical(line,i)

def check_numerical(line,index):
    if line[index]["word"] is not None:
        if line[index]["tag"] == 'CD' or line[index]["word"].lower() in number:
            base_index = line[index]["deps"]["nmod"][0]
            return line[base_index]["word"]

        else:
            return line[index]["word"]


def search_WP(line,head):
    for i in range(len(line)):
        if line[i]["head"] == head:
            if line[i]["tag"] == 'WP':
                return True
    return False



if __name__ == '__main__':
    num_tag = []
    sentences,num_tag = readFile('./verified_analogies.csv')
    lower_tie = 0
    upper_tie = 157
    start = time()
    count = 0
    base = []
    target = []
    like_count = 0
    text_output = "ID, Sentence, Target, Base\n"
    for i in range(lower_tie,upper_tie):
        print('____', i, '_____')
        print(sentences[i])
        t,b, like_count = dependency_parse(sentences[i],like_count)
        if t and b is not None:
            count+=1
        base.append(b)
        target.append(t)
        print(b, '________', t)
    print(len(base))
    print(len(target))
    print(like_count)
    for i in range(lower_tie,upper_tie):
        text_output +=  '"' + num_tag[i] + '","' +  sentences[i] + '","'+ str(base[i]) + '","' + str(target[i]) + '"' + "\n"
    writeTSVFile('base_target.csv', text_output)
    print('running time:', time() - start)
    print('detect:', count)

