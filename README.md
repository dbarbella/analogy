# Detecting Analogy in Text

### The Analogy Corpus
To build a model that can detect analogies in text, it is important that we have a corpus of analogies. This serves as a set of training examples from which our model can learn to classify whether a text contains an analogy or not.

The analogy corpus used for this project is in <file location>. It is built by searching in and extracting from the Brown corpus any sentence that contains an analogy indicator (the list of analogy indicators is specified in analogy_strings.py). The code to generate the corpus is in wordhunt.py.



### Naive Bayes Classifier
The analogy detection classifier is implemented in analogy_classifier.py.

To change the data set:
1. Change the root variable in personal.py to your local folder directory.
2. Make a text file which contains the analogies.
3. Specify its location in the analogy_file_name variable
4. Make a text file which contains the non-analogies.
5. Specify its location in the non_analogy_file_name variable

To test the classifier, write your test code under "# test classifier".
See methods you can used with NLTK's Naive Bayes Classifier at http://www.nltk.org/book/ch06.html.

If you can, document the test result with the data set you use in the classifier results spreadsheet in the research group folder.
