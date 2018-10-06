import os
import nltk
import csv
import random
import sys
import re
from time import time
from nltk import Tree
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget
from nltk.parse import stanford
os.chdir('..')
jar = './stanford-parser/jars/stanford-parser.jar'
model = './stanford-parser/jars/stanford-english-corenlp-2018-02-27-models.jar'
parser = stanford.StanfordParser(model,jar,encoding = 'utf8')

# this file  is used to draw tree and print tree images.

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

if __name__ == '__main__':
    sentences = readFile('./corpora/verified_analogies.csv')
    i = 0
    for s in sentences:
        parsed_sent = parser.raw_parse(s)
        for line in parsed_sent:
            cf = CanvasFrame()
            t = Tree.fromstring(str(line))
            tc = TreeWidget(cf.canvas(), t)
            cf.add_widget(tc, 10, 10)
            i += 1
            cf.print_to_file('./2018/treeImage/tree' + str(i) + '.ps')
            tree_name = './2018/treeImage/tree' + str(i) + '.ps'
            tree_new_name = './2018/treeImage/tree' + str(i) + '.png'
            os.system('convert ' + tree_name + ' ' + tree_new_name)
            cf.destroy()
        # for line in parsed_sent:
        #     for sentence in line:
        #         sentence.draw()
    print(sentences)
    # GUI
