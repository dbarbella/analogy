import os
import nltk
import csv
from nltk.parse import stanford
os.chdir('..')
jar = './stanford-parser/jars/stanford-parser.jar'
model = './stanford-parser/jars/stanford-english-corenlp-2018-02-27-models.jar'
parser = stanford.StanfordParser(model,jar,encoding = 'utf8')

# this file  is used to draw tree and print tree images.
class Node:
    def __init__(self,value):
        self.value = value
        self.parent = None
        self.children = []
        self.base = None
        self.target = None

    def createTree(self):
        if type(self.value[0]) == nltk.tree.Tree:
            for key in self.value:
                child = Node(key)
                child.parent = self
                subtree = child.createTree()
                self.children.append(subtree)
        return self
    def base_search(self,word,label):
        for key in self.children:
            if key.value._label == label and key.value[0][0] == word:
                return key
            if self.base is None:
                self.base = key.base_search(word,label)
        return self.base

    def target_search(self):
        if self.parent is not None:
            for child in self.parent.children:
                if child.value._label == "NP":
                    return child
            return self.parent.target_search()

def readFile(fileName):
    sent =[]
    with open(fileName) as file:
        readcsv = csv.reader(file, delimiter=',')
        for row in readcsv:
            sentence = row[1]
            tag = row[0]
            # num_tag.append(tag)
            sent.append(sentence)
    return sent

def search(sent,w,l):
    base_val, target_val = None,None
    parsed_sent = parser.raw_parse(sent)
    for line in parsed_sent:
        base,target = None,None
        root = Node(line[0])
        root.createTree()
        base = root.base_search(w,l)
        if base is not None:
            target = base.target_search()
            base_val = base.value
        if target is not None:
            target_val = target.value
    return base_val,target_val


if __name__ == '__main__':
    sentences = readFile('./corpora/verified_analogies.csv')
    count,like_count = 0,0

    for s in sentences:
        base, target = search(s,"like","PP")
        if base is not None and target is not None:
            count+=1
        else:
            print(s)
        words = s.split()
        if "like" in words:
            like_count+=1

    print("Number of found base and target: ",count)
    print("Number of the word 'like' in a sentence: ", like_count)

    # GUI
