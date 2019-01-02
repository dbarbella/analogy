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


def readFile(fileName):
    sent = []
    with open(fileName) as file:
        readcsv = csv.reader(file, delimiter=',')
        for row in readcsv:
            sentence = row[1]
            sent.append(sentence)
    return sent

def read_by_line(fileName):
    with open(fileName) as file:
        signals = file.readlines()
    signals = [x.strip() for x in signals]
    return signals

def writeCSVFile(text_output, to_dir):
    f = open(to_dir, 'w')
    for line in text_output:
            f.write(line)
    f.close()

def drawTreeAndSaveImage(s, i, dir):
    parsed_sent = parser.raw_parse(s)
    for line in parsed_sent:
        cf = CanvasFrame()
        t = Tree.fromstring(str(line))
        tc = TreeWidget(cf.canvas(), t)
        cf.add_widget(tc, 10, 10)
        i += 1
        cf.print_to_file(dir + str(i) + '.ps')
        tree_name = dir + str(i) + '.ps'
        tree_new_name = dir + str(i) + '.png'
        os.system('convert ' + tree_name + ' ' + tree_new_name)
        cf.destroy()