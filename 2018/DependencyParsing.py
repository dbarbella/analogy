from nltk.parse.stanford import StanfordDependencyParser
# from parser import readFile, writeTSVFile
from nltk.corpus import wordnet as wn

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
def dependency_parse(sentence):
    for v in tobe: #to be behaves oddly compared with other nouns, hence return it with the verb "behave", which makes the parser perform normal again
        sentence = sentence.replace('\b'+v+'\b','behave')
    result = dependency_parser.raw_parse(sentence)
    target = None
    base = None
    for line in result:
        target, tar_index = target_search(line.nodes)
        if tar_index is not None:
            base = base_search(tar_index,line.nodes)
    if base is not None:
        base = wn.morphy(changePronoun(base), wn.NOUN) #change the word back to a noun
        target = wn.morphy(changePronoun(target), wn.NOUN) #change back to a noun
        base = personName(base) #check if it is a person name
        target = personName(target)
        b = wn.synset(str(base)+ '.n.01')
        t = wn.synset(str(target)+ '.n.01')
        return {"base":base, "target":target, "similarity": wn.path_similarity(b,t), "sentence": sentence} #add features here
    else:
        return {"base": "", "target": "","similarity": 0.0, "sentence": sentence }

def personName(word):
    if word == None:
        return wn.morphy('person', wn.NOUN)
    return word

def changePronoun(word):
    if word.lower() == 'she':
        return 'female'
    elif word.lower() == 'he':
        return 'male'
    elif word.lower() == 'i' or word.lower() == 'you':
        return 'person'
    elif word.lower() == 'they'or word.lower() == 'we':
        return 'people'
    elif word.lower() == 'it' or word.lower == 'this' or word.lower()== 'that':
        return 'thing'
    elif word.lower() == 'these' or word.lower() == 'those':
        return 'things'
    else:
        return word.lower()

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

# if __name__ == '__main__':
#     num_tag = []
#     sentences,num_tag = readFile('./verified_analogies.csv')
#     lower_tie = 0
#     upper_tie = 157
#     start = time()
#     count = 0
#     base = []
#     target = []
#     like_count = 0
#     text_output = "ID, Sentence, Target, Base\n"
#     for i in range(lower_tie,upper_tie):
#         print('____', i, '_____')
#         print(sentences[i])
#         t,b = dependency_parse(sentences[i])
#         if t and b is not None:
#             count+=1
#         base.append(b)
#         target.append(t)
#         print(b, '________', t)
#     print(len(base))
#     print(len(target))
#     print(like_count)
#     for i in range(lower_tie,upper_tie):
#         text_output +=  '"' + num_tag[i] + '","' +  sentences[i] + '","'+ str(base[i]) + '","' + str(target[i]) + '"' + "\n"
#     writeTSVFile('base_target.csv', text_output)
#     print('running time:', time() - start)
#     print('detect:', count)

