Download Stanford parser at : https://nlp.stanford.edu/software/lex-parser.shtml and place it in a same folder as the parser.py file.
In the stanford-parser folder, create a folder named "jars" and place two files inside it:
- stanford-english-corenlp-2018-02-27-models.jar. The file can be found at 'Analogy Research/jars/'
- stanford parser.jar

In the folder 2018. Run the file: parser.py. This file has not yet been commented on. The file for now will print out the sentence, the base, and the target and the running time. Users are also welcomed to run "test_firstWeek.py" to see the result of the first week work.
The sample data is taken from sampleTraining.csv. There are only some samples as I am developing the algorithm.
To modify the batch, refer to variable lower_tie and upper_tie. The lower end min is 1 and the upper end maximum is 157

To run an arbitrary text, user needs to make a CSV file with this format:
"ID", "Sentence"
And replace the path to the CSV into the open file function. 
