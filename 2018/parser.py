import os
import nltk
# nltk.download()
import csv
# import en
import re
from time import time
from nltk.parse import stanford
jar = './stanford-parser/jars/stanford-parser.jar'
model = './stanford-parser/jars/stanford-english-corenlp-2018-02-27-models.jar'
parser = stanford.StanfordParser(model,jar,encoding = 'utf8')
verb = ['VB', 'VBZ', 'VBC' , 'VBN', 'VBP']
noun = ['NP','NN', 'NNP', 'NNS', 'PRP', 'S', 'SBAR']
pronoun =['UH', 'CC', ',', 'INTJ']
class Node:
    def __init__(self,value):
        self.value = value
        self.parent = None
        self.children = []

    def __ne__(self, other):
        return self.value == other.value

    def printTree(self):
        ret = ""
        for child in self.children:
            ret += "\t"+ str(child.parent) + "\n"
            ret += child.printTree()
        return ret

    def search_base(self,node):
        print(self.value)
        for child in node.children:
            self.search_base(child)
        return self.value

sentences = []
lower_tie = 1
upper_tie = 100

phrases = ['S','SBAR']

def linking_word_check(sent):
    # print(sent)
    linking_word = ["like","assss","similar","equivalent","similarly","equivalently","anagolous"]
    txt = nltk.word_tokenize(sent)
    list_of_word = nltk.pos_tag(txt)
    ind = 0
    for w in linking_word:
        if w in txt:
            ind = txt.index(w)
    # print(list_of_word[ind])
    # for w in list_of_word:
    #     if w[1] == 'JJ' or w[1] == 'RB':
    #         sent = sent.replace(w[0] + " ", "")
        # if w[1] == 'CC':
        #     sent = sent.replace(w[0], "")
            # print(w[0])
    print(sent)
    if list_of_word[ind][0].lower() == linking_word[2].lower() or list_of_word[ind][0] == linking_word[3] or list_of_word[ind][0] == linking_word[6]:
        # if list_of_word[ind][1] == "JJ":
        return parse(sent, list_of_word[ind][0], "JJ")
    elif list_of_word[ind][0].lower() == linking_word[0].lower() or list_of_word[ind][0] == linking_word[1]:
        if list_of_word[ind][1] == 'IN':
            return parse(sent,list_of_word[ind][0],"IN")

        if list_of_word[ind][1] == 'JJ':
            return parse(sent, list_of_word[ind][0], "JJ")
    elif list_of_word[ind][0] == linking_word[4] or list_of_word[ind][0] == linking_word[5]:
        return parse(sent, list_of_word[ind][0], "RB")

    else:
        return



def recursive_search(p,w,l,phrase, node):
    for i in range(len(p)):
        if type(p[i]) == nltk.Tree:
            if p[i]._label == l:
                if p[i][0] == w:
                    if len(p) > i + 1:
                        return p[i+1],node
            elif p[i]._label in phrases:
                for j in p[i]:
                    phrase,node = recursive_search(j,w,l,phrase,node)
            if len(p[i]) > 1:
                phrase,node = recursive_search(p[i],w,l,phrase, node)

    return phrase,node
def parse(sent,w,l):
    phrase = None
    base = []
    target = None
    parent_node = None
    parsed_sent = parser.raw_parse(sent)
    for line in parsed_sent:
        parent_node = Node(line)

        target, node =  recursive_search(line[0],w,l,phrase, parent_node)
        parent_node = creatingParentNode(line, parent_node)
        base = base_search2(target,parent_node,base)
    return base,target

def base_search2(target,parent_node,base):
    for child in parent_node.children:
        if str(child.value) == str(target):
            temp = child.parent
            p_child = {}
            result = []
            while temp.value._label not in noun:
                i = 0
                p_child[temp.parent] = temp
                temp = temp.parent
                # while temp.value[i]._label not in noun or temp.value[i]._label not in verb:
                #     if i == len(temp.value) - 1:
                #         break
                #     elif i <  len(temp.value) - 1:
                #         i+= 1
                while temp.value[i]._label in pronoun:
                    i += 1
                result.append(temp.value[i])

            f = []
            while len(result) > 0:
                f.append(result.pop())
            return f
        base = base_search2(target,child,base)
    return base


def segmentWords(s):
    return s.split()

def readFile(fileName):
    contents = []
    f = open(fileName)
    for line in f:
        contents.append(line)
    f.close()
    result = segmentWords('\n'.join(contents))
    return result


with open('./verified_analogies.csv') as file:
    readcsv = csv.reader(file,delimiter = ',')
    for row in readcsv:
        sentence = row[1]
        token = nltk.word_tokenize(sentence)
        sentences.append(sentence)

# def tryoutParent(sent):
#     parsed_sent = parser.raw_parse(sent)
#     node_parent = None
#     for line in parsed_sent:
#         p = Node(line)
#         node_parent = creatingParentNode(line,p)
#     return node_parent

def creatingParentNode(key, node):
    for count in range(len(key)):
        if len(key[count]) > 1:
            child = Node(key[count])
            child.parent = node
            subtree = creatingParentNode(key[count],child)
            node.children.append(subtree)
    return node




# print(len(sentences))
p_sent = []
start = time()
c = lower_tie
for i in range(lower_tie,upper_tie):
    p_sent.append(linking_word_check(sentences[i]))

for sent in p_sent:
    print("___", c + 1, "___")
    print("___",sent,"___")
    c += 1
end = time()
print("time: ", end - start)
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


# print(sentences)
# # GUI
# for line in parsed_sent:
#     for sentence in line:
#         sentence.draw()