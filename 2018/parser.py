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
class Node:
    def __init__(self,value):
        self.value = value
        self.parent = None
        self.children = []
    def __ne__(self, other):
        return self.value == other.value

    def search_base(self,node):
        print(self.value)
        for child in node.children:
            self.search_base(child)
        return self.value

sentences = []
lower_tie = 2
upper_tie = 6

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
    for w in list_of_word:
        if w[1] == 'JJ' or w[1] == 'RB':
            sent = sent.replace(w[0] + " ", "")
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
                        return p[i+1],p
            elif p[i]._label in phrases:
                for j in p[i]:
                    phrase,node = recursive_search(j,w,l,phrase,node)
            if len(p[i]) > 1:
                phrase,node = recursive_search(p[i],w,l,phrase, node)

    return phrase,node

def parse(sent,w,l):
    phrase = None
    base = None
    target = None
    node = None
    parsed_sent = parser.raw_parse(sent)
    for line in parsed_sent:
        target, node =  recursive_search(line[0],w,l,phrase, node)
        # print(node)
        base = base_search(node,line[0],base)
    return base,target

def base_search(target,p,base):
    for i in range(len(p)):
        if len(p[i])>1:
            for j in range(len(p[i])):
                if len(p[i][j])> 1:
                    if p[i][j] == target:
                        temp = j
                        while p[i][temp]._label not in verb:
                            if temp <= 0:
                                break
                            else:
                                temp = temp - 1
                        result = []
                        # for k in range(i):
                        #     result.append(p[k])
                        # # result.append(p[i-1])
                        # for l in range(temp):
                        #     result.append(p[i][l])
                        # return result
                        return [p[i-1],p[i][temp]]
                    base = base_search(target,p[i],base)
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

def delete_stopword(sent):
    stopword = set(readFile('./stopword.txt'))
    filtered_words = []
    for word in sent:
        if word not in stopword:
            filtered_words.append(word)
    return " ".join(filtered_words)

with open('./sampleTraining.csv') as file:
    readcsv = csv.reader(file,delimiter = ',')
    for row in readcsv:
        sentence = row[1]
        token = nltk.word_tokenize(sentence)
        sentence = delete_stopword(token)
        sentences.append(sentence)


print(len(sentences))
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