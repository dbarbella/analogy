1. Running the parsers
- Download Stanford parser at : https://nlp.stanford.edu/software/lex-parser.shtml and place it in the main directory.
In the stanford-parser folder, create a folder named "jars" and place two files inside it:
    - stanford-english-corenlp-2018-02-27-models.jar. The file can be found at 'Analogy Research/jars/'
stanford parser.jar
- To run the two parsers: In the folder 2018:
    - Syntactic Parser: Run the file: parser.py. The file will write a CSV with the sentence, and a boolean whether it can find the base and target or not. Users are welcomed to print out the base and target in the parser function
The sample data is taken from sampleTraining.csv. There are only some samples as I am developing the algorithm.
To modify the batch, refer to variable lower_tie and upper_tie. The lower end min is 1 and the upper end maximum is 157
    - Dependency Parser: run the file DependencyParsing.py. The file will write a CSV file with the sentence, the base, target, the similarity between base and target, a signal whether it can find the base and target, and the label of the sentence.
- To run an arbitrary text, user needs to make a CSV file with this format:
"ID", "Sentence"
And replace the path to the CSV into the open file function. 

2. Running the main code
- The main representation for now is "base_target", and the main classifier is "svm"
- In the main directory, run "main_interface.py". This will run over 100 trials of one corpus, consisting of parsed sentences by Syntactic Parser and Dependency Parser. 
    - It will output the maximum accuracy, f-score and confusion maxtrix of 100 trials
    - It will also output the maximum accuracy, minimum accuracy, and average accuracy over 100 trials
    - The users are welcome to explore more about the result by uncommenting the last 6 lines of the file
- Essential functions are stored in functions.py in main directory

