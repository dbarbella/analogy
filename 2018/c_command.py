import nltk
from time import time
import utilities

verbs = ["VB", "VBD", "VBG", "VBN", "VBZ", "VBP"]
nouns = ["DP", "NP", "NN", "NNS", "NNP", "NNPS", "PDT", "PRP"]
locations = ["in", "inside", "on", "onto", "into", "under", "above", "at", "from"]
instruments = ["by", "with"]


class Sentence:
    def __init__(self, sentence):
        self.sentence = sentence
        self.parsed_sent = utilities.parser.raw_parse(sentence)
        self.roles = {"action" : [],
                      "agent": [],
                      "theme": [],
                      "location": [],
                      "instrument": [],
                      "base": [],
                      "target": []}
        self.roles = self.search()

    def search(self, toReturn=None):
        for line in self.parsed_sent:
            root = Node(line[0])
            root.createTree()
            toReturn = root.themantic_search(self.roles)
        return toReturn


class Node:
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.children = []
        self.base = None
        self.target = None
        self.like_phrase = None
        self.word = ""

    def createTree(self):
        if type(self.value[0]) == nltk.tree.Tree:
            for key in self.value:
                child = Node(key)
                child.parent = self
                subtree = child.createTree()
                self.children.append(subtree)
        return self

    def search_verb(self, label, toReturn = None):
        for i in range(len(self.children)):
            if self.children[i].value._label in label:
                if i < len(self.children)-1:
                    if self.children[i+1].value._label != "RB":
                        return self.children[i].to_word()
                    elif self.children[i+1].value._label in verbs:
                        return self.children[i+1].to_word()

            toReturn = self.children[i].search_verb(label, toReturn)
        return toReturn

    def search_noun(self, label = nouns, toReturn = None):
        for key in self.children:
            if key.value._label in label:
                return key.to_word()
            if toReturn == None:
                toReturn = key.search_noun(label,toReturn)
        return toReturn

    def search_with_keywords(self, keyword, toReturn = None):
        for key in self.children:
            if key.value._label == "PP":
                if key.value[0][0] in keyword:
                    if len(key.children) > 1:
                        return key.children[1].to_word()
                elif key.value[0]._label == "ADVP" and key.value[1][0] in keyword:
                    return key.children[2].to_word()
            if toReturn == None:
                toReturn = key.search_with_keywords(keyword)
        return toReturn

    def search_up(self, label, toReturn=None):
        if self.parent is not None:
            for child in self.parent.children:
                if child.value._label in label:
                    return child.to_word()
                if toReturn is None:
                    toReturn = self.parent.search_up(label, toReturn)
        return toReturn

    def themantic_search(self, role):
        for key in self.children:
            if key.value._label == "VP":
                role["action"].append(key.search_verb(verbs))
                role["theme"].append(key.search_noun(nouns))
                role["agent"].append(key.search_up(nouns))
                role["location"].append(key.search_with_keywords(locations))
                role["instrument"].append(key.search_with_keywords(instruments))
                base,like_phrase = key.base_search("like","PP")
                if base is not None:
                    role["base"].append(base.to_word())
                    role["target"].append(like_phrase.target_search(base.value._label))
                return role
            role = key.themantic_search(role)
        return role

    def search_for_base_and_target(self, role, word, label):
        base = []
        target = []
        for key in self.children:
            if key.value._label == "VP":
                base,like_phrase = key.base_search(word,label)
                if base is not None:
                    role["base"].append(base.to_word())
                    role["target"].append(like_phrase.target_search(base.value._label))
                    role["action"].append(key.search_verb(verbs))
                return role
            role = key.search_for_base_and_target(role, word, label)
        return role

    def base_search(self, word, label, caches = []):
        for key in self.children:

            if key.value._label == label:
                if key.value[0][0] == word:
                    if len(key.children) > 1:
                        return key.children[1], key
                elif key.value[0]._label == "ADVP" and key.value[1][0] == word:
                    return key.children[2], key
            if self.base is None:
                self.base, self.like_phrase = key.base_search(word, label)
        return self.base, self.like_phrase

    def target_search(self, label):
        if self.parent is not None:
            for child in self.parent.children:
                if child.value._label == label:
                    child.to_word()
                    return child.word
                # elif child.value._label == ['S']:

            # if self.parent.value._label not in ["S", "SBAR"]:
            return self.parent.target_search(label)
        return None

    def to_word(self):
        if len(self.word) == 0:
            self.word = self._to_word("")
        
        return self.word

    def _to_word(self, temp):
        if type(self.value) == nltk.tree.Tree:
            if len(self.children) == 0:                
                return str(self.value[0])
            else:
                for child in self.children:

                    if type(child.value[0]) != nltk.tree.Tree:
                        temp += str(child.value[0]) + " "
                    else:
                        temp += child._to_word("")
        return temp


class Extract:
    def __init__(self):
        self.parsing_time = 0
        self.base_time = 0
        self.target_time = 0
        self.createTree_time = 0
    #sentence, word = "like", label
    #searches for all targets and their corresponding bases in a sentence.
    #also searches for action.
    def search(self, sent, w, l):
        role = {"base":[], "target":[], "action":[]}

        parsed_sent = utilities.parser.raw_parse(sent)
        for line in parsed_sent:
            root = Node(line[0])  # create the root node
            root.createTree()  # create a tree based on the parsed tree
            ###################
            role = root.search_for_base_and_target(role, w, l)
        return role["base"], role["target"], role["action"]


if __name__ == "__main__":
    signals = utilities.read_by_line("./2018/analogy_signals.txt")
    # sentences = utilities.readFile("./corpora/verified_analogies.csv")
    ext = Extract()

    output_text = "id, base, target, label\n"
    count  = 0
    sentences = ["a car that is like a real one is like a man in the pool."]

    for s in sentences:
        utilities.drawTreeAndSaveImage(s, 4, "./2018")
        if "like" in s:
            sent = Sentence(s)
            base = sent.roles['base']
            target = sent.roles['target']
            print("BASE: ", base, "TARGET: ", target)
            count += 1

        else:
            for signal in signals:
                if signal in s:
                    new_sent = s.replace(signal, "like")
                    sent = Sentence(new_sent)
                    base = sent.roles['base']
                    target = sent.roles['target']
                    print("BASE: ", base, "TARGET: ", target)
                    # if base == None:
                    #     base = []
                    # if target == None:
                    #     target = []
                    # output_text += str(count)+", "+"$".join(base)+", "+"$".join(target)+", 1\n"
                    # count += 1
                    break
        # utilities.writeCSVFile(output_text, "./outp.csv")