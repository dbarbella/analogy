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
    text = "I love cats. They are cute."
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

demo_test()
# draw_tree_save_image("This is my cat. It is my child.")
