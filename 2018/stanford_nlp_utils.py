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

text = "The cat sleeps."
client = CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner','parse','depparse','coref'], timeout=20000, memory='2G', be_quiet=False)
ann = client.annotate(text)


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


# draw_tree_save_image("This is my cat. It is my child.")
