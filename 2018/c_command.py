import os
import nltk
import csv
from time import time
from nltk.parse import stanford
from nltk import Tree
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget
os.chdir('..')
jar = './stanford-parser/jars/stanford-parser.jar'
model = './stanford-parser/jars/stanford-english-corenlp-2018-02-27-models.jar'
parser = stanford.StanfordParser(model,jar,encoding = 'utf8')

class Node:
    def __init__(self,value):
        self.value = value
        self.parent = None
        self.children = []
        self.base = None
        self.target = None
        self.like_phrase = None

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
            if key.value._label == label:
                if key.value[0][0] == word:
                    if len(key.children) > 1:
                        return key.children[1],key
                elif key.value[0]._label == "ADVP" and key.value[1][0] == word:
                    return key.children[2],key
            if self.base is None:
                self.base,self.like_phrase = key.base_search(word,label)
        return self.base,self.like_phrase

    def target_search(self):
        if self.parent is not None:
            for child in self.parent.children:
                if child.value._label == "NP":
                    return child
            return self.parent.target_search()

class Extract:
    def __init__(self):
        self.parsing_time = 0
        self.base_time = 0
        self.target_time = 0
        self.createTree_time = 0

    def readFile(self,fileName):
        sent =[]
        with open(fileName) as file:
            readcsv = csv.reader(file, delimiter=',')
            for row in readcsv:
                sentence = row[1]
                sent.append(sentence)
        return sent

    def search(self,sent,w,l):
        base_val, target_val, base, target = None,None,None,None
        parsed_sent = parser.raw_parse(sent)
        for line in parsed_sent:
            root = Node(line[0]) #create the root node
            root.createTree() #create a tree based on the parsed tree
            base,like_phrase = root.base_search(w,l) #search for base
            if base is not None: #if base is found then search for target
                target = like_phrase.target_search() #tranverse to the target from the base
                base_val = base.value #for returning
            if target is not None:
                target_val = target.value #for returning
        return base_val,target_val

    def read_by_line(self,fileName):
        with open(fileName) as file:
            signals = file.readlines()
        signals = [x.strip() for x in signals]
        return signals

    def writeCSVFile(self,text_output, to_dir):
        f = open(to_dir, 'w')
        for line in text_output:
            f.write(line)
        f.close()

    def drawTreeAndSaveImage(self,s,i):
        parsed_sent = parser.raw_parse(s)
        for line in parsed_sent:
            cf = CanvasFrame()
            t = Tree.fromstring(str(line))
            tc = TreeWidget(cf.canvas(), t)
            cf.add_widget(tc, 10, 10)
            i += 1
            cf.print_to_file('./2018/undetectedCases/tree' + str(i) + '.ps')
            tree_name = './2018/undetectedCases/tree' + str(i) + '.ps'
            tree_new_name = './2018/undetectedCases/tree' + str(i) + '.png'
            os.system('convert ' + tree_name + ' ' + tree_new_name)
            cf.destroy()

if __name__ == "__main__":
    extract = Extract()
    signals = extract.read_by_line("./2018/analogy_signals.txt")
    sentences = extract.readFile('./corpora/verified_analogies.csv')
    count,like_count,i = 0,0,0
    base_found = 0
    base, target = None,None
    for s in sentences:
        if "like" in s:
            base, target = extract.search(s,"like","PP")
        else:
            for signal in signals:
                if signal in s:
                    new_sent = s.replace(signal, "like")
                    base, target = extract.search(new_sent,"like", "PP")
                    break
        if base is not None and target is not None:
            count += 1
        # else:
        #     extract.drawTreeAndSaveImage(s,i)
        #     i+=1
    print("detected cases ",count)