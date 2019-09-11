# Designed to run with python 3.7.
# Run stanford_nlp_setup.py first


import stanfordnlp
import os
from stanfordnlp.server import CoreNLPClient

os.environ["CORENLP_HOME"] = "/Users/David/Documents/Code/analogy/corenlp"

# From the old way of doing it:
# parser = stanford.StanfordParser(model, jar, encoding='utf8')

# Check to make sure these are all correct
# Pretty sure they're not.
verbs = ["VB", "VBD", "VBG", "VBN", "VBZ", "VBP"]
nouns = ["DP", "NP", "NN", "NNS", "NNP", "NNPS", "PDT", "PRP"]
locations = ["in", "inside", "on", "onto", "into", "under", "above", "at", "from"]
instruments = ["by", "with"]


def demo_test(replace_like=False):
    # ['tokenize','ssplit','pos','lemma','ner','parse','depparse','coref']
    # text = "A cat in a cup is like a dog in a bucket."
    # text = "Rumor of a big battle spread like a grassfire up the valley." # This one doesn't parse correctly.
    # text = "When the sun came out, Stevie strode proudly into Orange Square," \
    #       "smiling like a landlord on industrious tenants. A cat is like a dog."
    text = ''
    text_big = '''and yet like a child among adults .
    I don't mean a few aesthetes who play about with sensations , like a young prince in a miniature dabbling his hand in a pool .
    Oh , he was being queer and careful , pawing about in the drawer and holding the bottle like a snake at the length of his arm .
    `` I went to the city And there I did Weep , Men a-crowing like asses , And living like sheep .
    Rumor of a big battle spread like a grassfire up the valley .
    When the sun came out , Stevie strode proudly into Orange Square , smiling like a landlord on industrious tenants .
    They gave the room a strange note of incongruity , like a mole on a beautiful face .
    It always came on , faithfully , just like a radio or juke box , whenever he started to worry too much about something , when the bad things tried to push their way into him .
    The design of a mechanical interlocking frame is much like a mechanical puzzle , but once understood , the principles can be applied to any track and signal arrangement .
    The sticks fell like a shower around her and she felt them sting her flesh and send tiny points of pain along her thighs .
    I saw the pony fall like a stone and the young warrior flew over its head , bouncing like a rubber ball .
    '''
    text += '''This dog is analogous to an atom.'''

    with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref'],
                       timeout=60000, memory='4G', be_quiet=True) as client:
        print("##########-----About to annotate...-----")
        # If we want to replace like, that needs to happen here. Let's not do that, however.
        ann = client.annotate(text)
        sen = ann.sentence[0]
        token = sen.token[0]
        print("*(((((")
        print(token.word)
        # sentence is a Sentence. Where and how is this defined?
        for sentence in ann.sentence:
            if replace_like:
                replace_with_like(sentence, signals, "like")
            for token in sentence.token:
                print(token.word, end=' ')
            print()

            constituency_parse = sentence.parseTree

            my_parse = CoreNLPNode(constituency_parse)
            my_parse.create_tree()
            my_parse.thematic_search()
            print("BASE: ", my_parse.roles["base"], "TARGET: ",
                  my_parse.roles["target"], "ACTION: ", my_parse.roles["action"])


# Figure out what i is.
# This is a unusual name for this; rename it, most likely.
def draw_tree_save_image(sentence):
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
            self.signals = read_by_line("analogy_signals.txt")
        else:
            self.root = root  # This one happens if we are setting up a child.
            self.signals = self.root.signals
        self.roles = {"action": [],
                      "agent": [],
                      "theme": [],
                      "location": [],
                      "instrument": [],
                      "base": [],
                      "target": []}

    def create_tree(self):
        for next_child in self.core_nlp_parse.child:
            child = CoreNLPNode(next_child, self.root)
            child.parent = self
            subtree = child.create_tree()
            self.children.append(subtree)
        return self

    def search_verb(self, label, to_return=None):
        for i in range(len(self.children)):
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
    def thematic_search(self, verbose=False):  # role):
        for child in self.children:
            if child.value == "VP":
                new_actions = child.search_verb(verbs)
                if verbose:
                    print("New Actions:", new_actions)
                if new_actions:
                    self.root.roles["action"].append(new_actions)

                new_themes = child.search_noun(nouns)
                if verbose:
                    print("New Actions:", new_themes)
                if new_themes:
                    self.root.roles["theme"].append(new_themes)

                new_agents = child.search_up(nouns)
                if verbose:
                    print("New Agents:", new_agents)
                if new_agents:
                    self.root.roles["agent"].append(new_agents)

                new_locations = child.search_with_keywords(locations)
                if verbose:
                    print("New Locations:", new_locations)
                if new_locations:
                    self.root.roles["location"].append(new_locations)

                new_instruments = child.search_with_keywords(instruments)
                if verbose:
                    print("New Instruments:", new_instruments)
                if new_instruments:
                    self.root.roles["instrument"].append(new_instruments)

                base, like_phrase = child.base_search(self.signals, "PP", verbose=True)
                if verbose:
                    print("Base:", base, like_phrase)
                if base:
                    if verbose:
                        print(base.to_word())
                        print(like_phrase.to_word())
                    self.root.roles["base"].append(base.to_word())
                    target = like_phrase.target_search(base.value)
                    self.root.roles["target"].append(target)
                else:
                    if verbose:
                        print("Returned None for Base")
            child.thematic_search()

    # Word is the word "like" in how we're typically using this.
    # Label is PP. Neither of these things ever change.
    # The original version of this returns a node consider revisiting that.
    def base_search(self, signals, label, verbose=False):
        if verbose:
            print("starting base_search with", self.value, signals, label)
        # For each of the children of the current node, do the following:
        for child in self.children:
            if verbose:
                print("next child's value:", child.value)
            # Check to see if that child is a PP.
            if child.value == label:
                if verbose:
                    print("That was equal to label")
                # Check to see if the PP's leftmost grandchild is the word "like."
                print(self.value)
                leftmost_grandchild_value = child.children[0].children[0].value
                if verbose:
                    print("leftmost_grandchild_value is", leftmost_grandchild_value)
                if leftmost_grandchild_value in signals:
                    if len(child.children) > 1:
                        if verbose:
                            print("Returning from top case.")
                            print("child:")
                            print(child.core_nlp_parse)
                            print("child's Second child's second child:")
                            print(child.children[1].children[1].core_nlp_parse)
                        base_found = child.children[1]
                        like_phrase_found = child
                        return base_found, like_phrase_found
                elif child.value[0] == "ADVP" and child.value[1][0] in signals:
                    if verbose:
                        print("Returning from second case.")
                    # This needs revisiting, it's currently nothing.
                    return child.children[2], child
            if self.base is None:
                self.base, self.like_phrase = child.base_search(signals, label, verbose=verbose)
        return self.base, self.like_phrase

    def target_search(self, label):
        if self.parent is not None:
            for child in self.parent.children:
                if child.value == label:
                    child.to_word()
                    return child.word
                # elif child.value._label == ['S']:

            # if self.parent.value._label not in ["S", "SBAR"]:
            return self.parent.target_search(label)
        return None

    # The purpose of this is to avoid redoing calculations
    def to_word(self):
        if self:
            if len(self.word) == 0:
                self.word = self._to_word("")[:-1]  # Cut off the final space.
            return self.word

    def _to_word(self, temp):
        if len(self.children) == 0:
            return self.value + " "
        else:
            for child in self.children:
                temp += child._to_word("")
        return temp


def read_by_line(file_name):
    with open(file_name) as file:
        signals = file.readlines()
    signals = [x.strip() for x in signals]
    return signals


# This is a hack; figure out something better.
# This needs to come before we tokenize it and do the other proecessing.
def replace_with_like(sentence, signals, ANALOGY_KEYWORD):
    if "like" not in sentence.word:  # Figure out the right way to check this.
        print("Replacing Like")
        for signal in signals:
            if signal in sentence:
                sentence = sentence.replace(signal, ANALOGY_KEYWORD)


if __name__ == "__main__":
    demo_test(replace_like=False)
