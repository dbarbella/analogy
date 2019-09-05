# Designed to run with python 3.7.
# Run stanford_nlp_setup.py first


import stanfordnlp
import os
from stanfordnlp.server import CoreNLPClient

os.environ["CORENLP_HOME"] = "/Users/David/Documents/Code/analogy/corenlp"

'''
doc = pipeline("Barack Obama was born in Hawaii.  He was elected president in 2008.")
doc.sentences[0].print_dependencies()
'''

# From the old way of doing it:
# parser = stanford.StanfordParser(model, jar, encoding='utf8')

# Check to make sure these are all correct
# Pretty sure they're not.
verbs = ["VB", "VBD", "VBG", "VBN", "VBZ", "VBP"]
nouns = ["DP", "NP", "NN", "NNS", "NNP", "NNPS", "PDT", "PRP"]
locations = ["in", "inside", "on", "onto", "into", "under", "above", "at", "from"]
instruments = ["by", "with"]

def demo_test():
    # ['tokenize','ssplit','pos','lemma','ner','parse','depparse','coref']
    # text = "A cat in a cup is like a dog in a bucket."
    text = "A cat is like a dog."
    with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref'],
                       timeout=60000, memory='4G', be_quiet=True) as client:
        #client = CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos'], timeout=20000, memory='2G', be_quiet=False)
        print("##########-----About to annotate...-----")
        # ann = client.annotate(text, annotators=['tokenize','ssplit','pos','lemma','ner','parse','depparse','coref'])
        ann = client.annotate(text)
        sentence = ann.sentence[0]

        print('---')
        print('Constituency parse of first sentence')
        constituency_parse = sentence.parseTree

        '''
        print(constituency_parse)
        print(constituency_parse.value)
        print("$$")

        print('---')
        print('first subtree of constituency parse')
        print(constituency_parse.child[0])
        print(constituency_parse.child[0].value)
        print("@@")

        print('---')
        print('Number of subtrees of constituency parse')
        print(len(constituency_parse.child))

        print("Roles:")
        print(test_thematic_search(constituency_parse))
        '''

        my_parse = CoreNLPNode(constituency_parse)
        my_parse.create_tree()
        my_parse.thematic_search()
        print("Final roles:", my_parse.roles)


def test_thematic_search(parse):
    # This might need to go up top, with the call underneath, and then we don't make
    # roles an argument.
    roles = {"action": [],
             "agent": [],
             "theme": [],
             "location": [],
             "instrument": [],
             "base": [],
             "target": []}

    # This is sketchy, and probably belongs in a class.
    def test_thematic_search_find(parse):
        for child in parse.child:
            print("Next child value:", child)
            if child.value == "VP":
                # So here we need to replace key.search_verb with something that does search_verb but with child.
                roles["action"].append(search_verb(child, verbs))
                roles["theme"].append(search_noun(child, nouns))
                # roles["agent"].append(search_up(child, nouns))
                '''
                roles["location"].append(child.search_with_keywords(locations))
                roles["instrument"].append(child.search_with_keywords(instruments))
                '''
                # base, like_phrase = key.base_search("like", "PP")
                # if base is not None:
                #    roles["base"].append(base.to_word())
                #    roles["target"].append(like_phrase.target_search(base.value._label))
                # return roles
            test_thematic_search_find(child)
            print("Roles now:", roles)
        # return roles
    test_thematic_search_find(parse)
    return(roles)

'''
# It's very unclear why this is doing what it's doing.
def search_verb(node, verbs, to_return=None):
    print("Calling search_verb", node.value, verbs)
    for i in range(len(node.child)):
        print("Next subchild: ", node.child[i].value)
        if node.child[i].value in verbs:
            if i < len(node.child) - 1:
                if node.child[i + 1].value != "RB":
                    # This needs to get the word at the location
                    print("returning node.child[i].value:", node.child[i].value)
                    return node.child[i].value
                elif node.child[i + 1].value in verbs:
                    print("returning node.child[i + 1].value:", node.child[i + 1].value)
                    return node.child[i + 1].value
        else:
            print("That wasn't in verbs.")

        to_return = node.child[i].search_verb(label, to_return)
    return to_return


def search_noun(node, nouns, to_return=None):
    for next_child in node.child:
        if next_child.value in nouns:
            return next_child.value
        if to_return is None:
            to_return = search_noun(next_child, nouns, to_return)
    return to_return


# For this one to work, we need to be able to go up the tree. Are any others like that?
# If that's the case, then we need to rebuild the tree after all.
# Yeah, we're going to need to rebuild the tree with links up.
def search_up(node, label, to_return=None):
    if self.parent is not None:
        for child in node.parent.child:  # Not sure if parent works here; might need another strategy
            if child.value in label:
                return child.to_word()
            if to_return is None:
                to_return = node.parent.search_up(label, to_return)
    return to_return

# May no longer be used?
def to_word(self):
    if len(self.word) == 0:
        self.word = self._to_word("")

    return self.word

'''

# Figure out what i is.
# This is a unusual name for this; rename it, most likely.
def draw_tree_save_image(sentence):  #, i, dir):
    pipeline = stanfordnlp.Pipeline(models_dir="../stanfordnlp_resources")  # This sets up a default neural pipeline in English
    doc = pipeline(sentence)
    doc.sentences[0].print_dependencies()

    # This used nltk's parser in the old way. What's the right way to do this with stanfordnlp?
    # parsed_sent = parser.raw_parse(sentence)

    '''
    # This is the part that actually draws
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
    '''


class CoreNLPNode:
    def __init__(self, core_nlp_parse, root=None):
        self.core_nlp_parse = core_nlp_parse
        self.value = core_nlp_parse.value
        self.parent = None
        self.children = []
        self.base = None
        self.target = None
        self.like_phrase = None
        self.word = ""
        if not root:
            self.root = self  # This happens if we are setting up the root.
        else:
            self.root = root  # This one happens if we are setting up a child.
        self.roles = {"action": [],
                      "agent": [],
                      "theme": [],
                      "location": [],
                      "instrument": [],
                      "base": [],
                      "target": []}

    def create_tree(self):
        # if type(self.value[0]) == nltk.tree.Tree:
        for next_child in self.core_nlp_parse.child:
            child = CoreNLPNode(next_child, self.root)
            child.parent = self
            subtree = child.create_tree()
            self.children.append(subtree)
        return self

    def search_verb(self, label, to_return=None):
        for i in range(len(self.children)):
            print(self.children[i].value)
            if self.children[i].value in label:
                if i < len(self.children) - 1:
                    if self.children[i + 1].value != "RB":
                        return self.children[i].to_word()
                    elif self.children[i + 1].value in verbs:
                        return self.children[i + 1].to_word()
            to_return = self.children[i].search_verb(label, to_return)
        return to_return

    def search_noun(self, label=nouns, to_return=None):
        for key in self.children:
            if key.value in label:
                return key.to_word()
            if to_return is None:
                to_return = key.search_noun(label, to_return)
        return to_return

    def search_with_keywords(self, keyword, to_return=None):
        for key in self.children:
            if key.value == "PP":
                if key.value[0][0] in keyword:
                    if len(key.children) > 1:
                        return key.children[1].to_word()
                elif key.value[0] == "ADVP" and key.value[1][0] in keyword:
                    return key.children[2].to_word()
            if to_return is None:
                to_return = key.search_with_keywords(keyword)
        return to_return

    def search_up(self, label, to_return=None):
        if self.parent is not None:
            for child in self.parent.children:
                if child.value in label:
                    return child.to_word()
                if to_return is None:
                    to_return = self.parent.search_up(label, to_return)
        return to_return

    # The other version of this takes role as an argument. Is that somehow important?
    # What may be going on is that we're changing things for the wrong self.
    def thematic_search(self): # role):
        for child in self.children:
            if child.value == "VP":
                new_actions = child.search_verb(verbs)
                print("New Actions:", new_actions)
                self.root.roles["action"].append(new_actions)

                new_themes = child.search_noun(nouns)
                print("New Actions:", new_themes)
                self.root.roles["theme"].append(new_themes)

                new_agents = child.search_up(nouns)
                print("New Agents:", new_agents)
                self.root.roles["agent"].append(new_agents)

                new_locations = child.search_with_keywords(locations)
                print("New Locations:", new_locations)
                self.root.roles["location"].append(new_locations)

                new_instruments = child.search_with_keywords(instruments)
                print("New Instruments:", new_instruments)
                self.root.roles["instrument"].append(new_instruments)

                base, like_phrase = child.base_search("like", "PP")
                print("Base:", base, like_phrase)
                print(base.value)
                print(like_phrase.value)
                #self.root.roles["base"].append(base.to_word())
                '''
                if base is not None:
                    self.root.roles["base"].append(base.to_word())
                    target = like_phrase.target_search(base.value)
                    self.root.roles["target"].append(target)
                # return roles
                '''
            child.thematic_search()


    def search_for_base_and_target(self, role, word, label):
        base = []
        target = []
        for child in self.children:
            if child.value == "VP":
                base, like_phrase = child.base_search(word, label)
                if base is not None:
                    role["base"].append(base.to_word())
                    role["target"].append(like_phrase.target_search(base.value._label))
                    role["action"].append(child.search_verb(verbs))
                return role
            role = child.search_for_base_and_target(role, word, label)
        return role

    def base_search(self, word, label, caches=[]):
        print("starting base_search with", self.value, word, label)
        for child in self.children:
            print("next child's value:", child.value)
            if child.value == label:
                print("That was equal to label")
                print("@@@@@child.value[0][0] is", child.value[0][0])
                # Unclear what this is for, because nothing is documented, but this should be checking for like.
                # Instead, it's currently getting the first letter of the label.
                # This is really hacked together.
                if True:  # child.value[0][0] == word:  # This isn't right, this is just to see what happens.
                    if len(child.children) > 1:
                        print("Returning from top case.")
                        print(child.core_nlp_parse)
                        return child.children[1], child
                elif child.value[0] == "ADVP" and child.value[1][0] == word:
                    print("Returning from second case.")
                    return child.children[2], child
            if self.base is None:
                self.base, self.like_phrase = child.base_search(word, label)
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

    '''
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
    '''
    def _to_word(self, temp):
        if len(self.children) == 0:
            return self.value
        else:
            for child in self.children:
                temp += child._to_word("")
        print("_to_word returning:", temp)
        return temp


demo_test()
# draw_tree_save_image("This is my cat. It is my child.")
