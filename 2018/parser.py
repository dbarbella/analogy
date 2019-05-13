import os
import nltk
import csv
import random
import sys
import re
from time import time
from nltk.parse import stanford
os.chdir('..')
jar = './stanford-parser/jars/stanford-parser.jar'
model = './stanford-parser/jars/stanford-english-corenlp-2018-02-27-models.jar'
parser = stanford.StanfordParser(model,jar,encoding = 'utf8')

verb = ['VB', 'VBZ', 'VBC' , 'VBN', 'VBP']
noun = ['NP','NN', 'NNP', 'NNS', 'PRP', 'S', 'WHNP']
pronoun =['UH', 'CC', ',', 'INTJ','ADVP','RRC','PRN','RB', 'TO']
phrases = ['S','SBAR']
class Node:
    def __init__(self,value):
        self.value = value
        self.parent = None
        self.children = []

    def __ne__(self, other):
        return self.value == other.value

    def printTree(self,layer):
        ret = ""
        layer += 1
        for child in self.children:
            ret += str(layer) + "\t" * layer + str(child.parent.value) + "\n"
            ret += child.printTree(layer)
        return ret

sentences = []
lower_tie = 4
upper_tie = 157

base_time, tartget_time, createTree_time, parsing_time = 0, 0, 0, 0

class Grammar_parse:
    def __init__(self):
        self.parsing_time = 0
        self.base_time = 0
        self.target_time = 0
        self.createTree_time = 0
    def linking_word_check(self,sent):
        # print(sent)
        linking_word = ["like","assss","similar","equivalent","similarly","equivalently","anagolous"]
        txt = nltk.word_tokenize(sent)
        list_of_word = nltk.pos_tag(txt)
        ind = 0
        for w in linking_word:
            if w in txt:
                ind = txt.index(w)
        if list_of_word[ind][0].lower() == linking_word[0].lower() or list_of_word[ind][0] == linking_word[1]:
            if list_of_word[ind][1] == 'IN':
                base,target = self.parse(sent,list_of_word[ind][0],"IN")
                return base, target

            if list_of_word[ind][1] == 'JJ':
                base,target = self.parse(sent, list_of_word[ind][0], "JJ")
                return base, target
            else:
                return None,None

        elif list_of_word[ind][0] == linking_word[4] or list_of_word[ind][0] == linking_word[5]:
            base,target = self.parse(sent, list_of_word[ind][0], "RB")
            return base, target
        else:
            return None,None

    def recursive_search(self,p,w,l,phrase, node):
        for i in range(len(p)):
            if type(p[i]) == nltk.Tree:
                if p[i]._label == l:
                    if p[i][0] == w:
                        if len(p) > i + 1:
                            return p[i+1],p
                elif p[i]._label in phrases:
                    for j in p[i]:
                        phrase,node = self.recursive_search(j,w,l,phrase,node)
                if len(p[i]) > 1:
                    phrase,node = self.recursive_search(p[i],w,l,phrase, node)
        return phrase,node

    def parse(self,sent,w,l):
        phrase = None
        base = None
        target = None
        start1 = time()
        parsed_sent = parser.raw_parse(sent)
        start2 = time()
        self.parsing_time += start2 - start1
        for line in parsed_sent:
            parent_node = Node(line)
            target, node =  self.recursive_search(line[0],w,l,phrase, parent_node)
            start3 = time()
            self.base_time += start3 - start2
            parent_node = self.creatingParentNode(line, parent_node)
            start4 = time()
            self.createTree_time = start4 - start3
            base = self.base_search2(target,parent_node,base)
            start5 = time()
            self.target_time += start5 - start4
        return base,target

    def base_search2(self,target,parent_node,base):
        for child in parent_node.children:
            if str(child.value) == str(target):
                temp = child.parent
                p_child = {}
                result = []

                while temp is not None and temp.value._label not in noun :
                    p_child[temp.parent] = temp
                    temp = temp.parent

                for temp in p_child:
                    i = 0
                    childNode = p_child[temp]
                    while temp is not None and (temp.value[i]) != (childNode.value) and i < len(temp.value) - 1 :
                        if temp.value[i]._label not in pronoun:
                            result.append(temp.value[i])
                            # print("temp\n",temp.value)
                        i+=1
                subject = None
                if len(result) > 0:
                    firstVerb = None
                    subject = result.pop()
                    if type(subject[0]) == nltk.tree.Tree:
                        # print(subject[0]._label)
                        if len(result)>0:
                            if result[len(result)-1]._label in verb:
                                firstVerb = result.pop()
                            if subject[0]._label == 'PRP':
                                f = []
                                f.append(subject)
                                f.append(firstVerb)
                                f += result
                                return f
                            else:
                                return subject
                return subject
            base = self.base_search2(target,child,base)
        return base

    def creatingParentNode(self,key, node):
        for count in range(len(key)):
            if type(key[count]) == nltk.tree.Tree:
                child = Node(key[count])
                child.parent = node
                subtree = self.creatingParentNode(key[count],child)
                node.children.append(subtree)
        return node

    def checkAvailabilityNode(self,key):
        count = len(key)
        for i in range(count):
            if len(key[i]) > 1:
                return True
        return False
    def segmentWords(self,s):
        return s.split()

def readFile(fileName):
    sent =[]
    with open(fileName) as file:
        readcsv = csv.reader(file, delimiter=',')
        for row in readcsv:
            sentence = row[1]
            sent.append(sentence)
    return sent

base = []
target = []
trash = [",","'", '"', "`"]

def chunk(sentence):
    b, t = linking_word_check(sentence)
    for s in trash:
        b = "".join(str(b).split(s))
        t = "".join(str(t).split(s))
    return b,t
def writeCSVFile(text_output, to_dir):
    f = open(to_dir, 'w')
    for line in text_output:
            f.write(line)
    f.close()

if __name__ == '__main__':
    # pos = readFile('./corpora/verified_analogies.csv')
    # neg = readFile('./corpora/verified_non_analogies.csv')
    # samples = pos + neg
    samples = readFile('./corpora/verified_analogies.csv')
    random.seed(1234)
    random.shuffle(samples)
    txt = ""

    count = 0
    grammar_parse = Grammar_parse()
    for s in samples:
        base,target, = grammar_parse.linking_word_check(s)
        if base and target is not None:
            count +=1
            # print(str(base) + "___"+ str(target))
            # if base is not None and target is not None:
            #     txt += '"' + samples[i] + '","' + str(1) + '"\n'
            # else:
            #     txt += '"' + samples[i] + '","' + str(0) + '"\n'
    print("parsing time: ", grammar_parse.parsing_time)
    print("base time: ", grammar_parse.base_time)
    print("target time: ", grammar_parse.target_time)
    print("create tree time: ", grammar_parse.createTree_time)

    # writeCSVFile(txt, './random_parse_tree.csv')
# for i in range(lower_tie,upper_tie):
#     p_sent.append(tryoutParent(sentences[i]))
# for p in p_sent:
#     print(p.printTree())
# c=0
# new_sent = sentences[lower_tie:upper_tie]
# parsed_sent = parser.raw_parse_sents(new_sent)
# for line in parsed_sent:
#     for sentence in line:
#         print("___",c+1,"____")
#         # sentence.chomsky_normal_form()
#         print(sentence,end = "\n\n")
#         c+=1

# sentences = readFile('./verified_analogies.csv')
# for s in sentences:
#     chunk(s)
# print(sentences)
# # GUI
# for line in parsed_sent:
#     for sentence in line:
#         sentence.draw()