import nltk
from nltk.parse.stanford import StanfordDependencyParser
# from parser import readFile, writeTSVFile
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
import json
path_to_jar = './stanford-parser/jars/stanford-parser.jar'
path_to_models_jar = './stanford-parser/jars/stanford-english-corenlp-2018-02-27-models.jar'
from time import time
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
verb = ['VB', 'VBZ', 'VBC' , 'VBN', 'VBP', 'root', 'VBG', 'VBD']
subject = ['nsubjpass', 'nsubj']
noun = ['NP','NN', 'NNP', 'NNS', 'PRP']
linking_words = ["like"]
number = ["tens", "hundreds", "thousands", "millions", "billions", "trillions","dose","dozen","piece","fragment"]
tobe = ["was being", "were being", "will have been", "will be", "is going to", "am going to", "are going to", "has been", "have been", "am", "are", "is", "was", "were"]

def parse(sentence):
    temp = ""
    for v in tobe: #to be behaves oddly compared with other nouns, hence return it with the verb "behave", which makes the parser perform normal again
        temp = sentence.replace('\b'+v+'\b','behave')
    result = dependency_parser.raw_parse(temp)
    for line in result:
        return toDict(line.nodes,sentence),line

def dependency_parse(result,sentence,label):
    base = None
    target, tar_index = target_search(result.nodes)
    if tar_index is not None:
        base = base_search(tar_index,result.nodes)
    if base is not None and len(target) > 1:
        base, b = changePronoun(base) #check if it is a person name
        print(sentence, "---", base, target)
        target, t = changePronoun(target)
        similarity = wn.path_similarity(b,t)
        if similarity is None:
            similarity = 0.0
        return {"base":base, "target":target, "similarity": similarity, "sentence": sentence, "detected": True, "label": label} #add features here
    else:
        return {"base": "", "target": "","similarity": 0.0, "sentence": sentence, "detected": False, "label": label}




def changePronoun(word):
    tag = nltk.pos_tag(word_tokenize((word)))
    if tag[0][1] == 'NN' or tag[0][1]== 'NNS':
        w= wn.morphy(word, wn.NOUN)
        if w is None:
            w = wn.morphy('person', wn.NOUN)
            return w, wn.synset(str(w) + '.n.01')
        else:
            return w, wn.synset(str(w) + '.n.01')
    elif tag[0][1]== 'NNP' or tag[0][1]== 'NNPS':
        w = wn.morphy('person', wn.NOUN)
        return w, wn.synset(str(w) + '.n.01')
    elif tag[0][1]== 'PRP' or tag[0][1]== 'PRPS':
        if word.lower() == 'she':
            w = wn.morphy('female', wn.NOUN)
            return w, wn.synset(str(w) + '.n.01')
        elif word.lower() == 'he':
            w = wn.morphy('male', wn.NOUN)
            return w, wn.synset(str(w) + '.n.01')
        elif word.lower() == 'i' or word.lower() == 'you' or word.lower() == 'who':
            w = wn.morphy('person', wn.NOUN)
            return w, wn.synset(str(w) + '.n.01')
        elif word.lower() == 'they'or word.lower() == 'we':
            w = wn.morphy('people', wn.NOUN)
            return w, wn.synset(str(w) + '.n.01')
        elif word.lower() == 'it':
            w = wn.morphy('thing', wn.NOUN)
            return w, wn.synset(str(w) + '.n.01')
    elif tag[0][1]== 'DT':
        if  word.lower() == 'this' or word.lower()== 'that':
            w = wn.morphy('thing', wn.NOUN)
            return w, wn.synset(str(w) + '.n.01')
        elif word.lower() == 'these' or word.lower() == 'those':
            w = wn.morphy('things', wn.NOUN)
            return w, wn.synset(str(w) + '.n.01')
    elif tag[0][1]== 'WRP' or tag[0][1]== 'WRT':
        w = wn.morphy('thing', wn.NOUN)
        return w, wn.synset(str(w) + '.n.01')
    elif '-' in word:
        ind = word.index('-')
        w = wn.morphy(word[:ind], wn.NOUN)
        return w, wn.synset(str(w) + '.n.01')
    elif tag[0][1]in verb:
        w = wn.morphy(word, wn.VERB)
        return w, wn.synset(str(w) + '.v.01')
    elif tag[0][1]== 'JJ' or tag[0][1]== 'JJS':
        w = wn.morphy(word, wn.ADJ)
        if w == None:
            w= wn.morphy('person', wn.NOUN)
            return w, wn.synset(str(w) + '.n.01')
        else:
            return w, wn.synset(str(w) + '.a.01')
    else:
        w = wn.morphy('thing', wn.NOUN)
        return w, wn.synset(str(w) + '.n.01')

def target_search(p):
    for i in range(len(p)):
        if p[i]["word"] in linking_words:
            index = p[i]["head"]

            return check_numerical(p,index), index
    return None, None

def base_search(tar_index,line):
    base_index = tar_index
    if base_index == 0: #there is no base
        return None
    grand_base_index = line[base_index]["head"] # the head of current index
    if line[grand_base_index]["tag"] in noun and line[grand_base_index]["rel"] == 'dobj': #check if the head of the current index is a noun and is a direct object
        for i in range(grand_base_index,tar_index):
            if "compound" in line[i]["deps"] and line[i]["head"] == grand_base_index: #this case deals with V-ing sentences
                return check_numerical(line,grand_base_index)
    while line[base_index]["tag"] not in verb: #find the closest verb to the target
        temp = line[base_index]["head"]
        if temp == 0:
            break
        else:
            base_index = temp
    grand_base_index = line[base_index]["head"]
    while line[grand_base_index]["tag"] in verb:#trace back to the main verb in a multi-verb sentence
        for s in subject: #find the subject of this verb
            if s in line[base_index]["deps"]:
                return find_subject(line,base_index,s)
        for s in subject:
            if s in line[grand_base_index]["deps"]:
                return find_subject(line,grand_base_index,s)
        base_index = line[base_index]["head"]
        grand_base_index = line[base_index]["head"]

    if search_WP(line,base_index): #if there is a Wh-phrase, return the Noun that the Wh is refering to
        return check_numerical(line,grand_base_index)
    else:
        for s in subject: #up to this stage, it is most likely that the structure would be like this: S V like O
            if s in line[base_index]["deps"]:
                return find_subject(line,base_index,s)

def find_subject(line,index,s):
    """find the subject based on the verb"""
    b = line[index]["deps"][s]
    temp = b[len(b) - 1]
    return check_numerical(line, temp)

def check_numerical(line,index):
    """check if the current noun is a number noun, if it is, return the real noun"""
    if line[index]["word"] is not None:
        if line[index]["tag"] == 'CD' or line[index]["word"].lower() in number:
            if "nmod" in line[index]["deps"]:
                temp = line[index]["deps"]["nmod"]
                base_index = temp[len(temp)-1]
                return line[base_index]["word"]
            else:
                return line[index]["word"]
        else:
            return line[index]["word"]


def search_WP(line,head):
    """search if the current sentence has a Wh structure, this kinda partly solves the anaphora but not entirely"""
    for i in range(len(line)):
        if line[i]["head"] == head:
            if line[i]["tag"] == 'WP':
                return True
    return False

def writeCSVFile(text_output, to_dir):
    f = open(to_dir, 'w')
    for line in text_output:
            f.write(line)
    f.close()


def toDict(line,sent):
    dic = {}
    properties = ["address", "ctag", "feats", "head", "lemma", "rel", "tag", "word"]
    for i in range(len(line)):
        temp = {}
        for p in properties:
            temp[p] = line[i][p]
        deps = {}
        for key in line[i]["deps"]:
            deps[key] = line[i]["deps"][key]
        temp["deps"] = deps
        dic[i] = temp
    return dic

