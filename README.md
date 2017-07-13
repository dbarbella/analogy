# Detecting Analogy in Text

### The Demo Analogy Corpus
To build a model that can detect analogies in text, it is important that we have a starting corpus of analogies. This serves as a set of training examples from which our model can learn to classify whether a text contains an analogy or not.

The analogy corpus used for this project is in corpora/analogy_sentences.txt. It is built by searching in and extracting from the Brown corpus any sentence that contains an analogy indicator (the list of analogy indicators is specified in analogy_strings.py). The code to generate the corpus is in wordhunt.py.



### Naive Bayes Classifier
The analogy detection classifier is implemented in analogy_classifier.py.

For the classifier to accurately detect whether a text contains an analogy, first it must be trained with two sets of data. The first set is a list of sentences that contain analogies and the other is a list of sentences with non-analogies. The list of analogies is an analogy corpus generated with the method documented in the previous section.

To make the classification more as accurate as possible, every training example used to train the classifier is represented in a different format. This representation is a combination by the analogy indicating words or phrases the text contains, along with the part-of-speech (POS) tags of those words/phrases. For example, the sentence "It is like handing a loaded automatic to an 8-year-old." is represented by "(('like', 'IN')" because here the word "like" signifies that the sentence may have an analogy and it serves as an IN (the POS tag for conjunction) in this sentence. The list of POS tags can be viewed at http://www.comp.leeds.ac.uk/amalgam/tagsets/upenn.html.

To change the data set used to train the classifier:
1. Change the root variable in personal.py to your local folder directory.
2. Make a text file which contains the analogies.
3. Specify its location in the analogy_file_name variable
4. Make a text file which contains the non-analogies.
5. Specify its location in the non_analogy_file_name variable

To test the classifier, write your test code under "# test classifier".
See methods you can use with NLTK's Naive Bayes Classifier at http://www.nltk.org/book/ch06.html.

The test results are documented in the classifier results spreadsheet in the research group folder.
