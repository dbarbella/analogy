from __future__ import print_function
import functions
import helpers
import parameters_file
#--------
#from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

def main():
    positive_set = 'test_extractions/bc_samples.txt' #'test_extractions/test-neural-hash-samples.txt' 
    negative_set = 'test_extractions/bc_grounds.txt' #'test_extractions/test-neural-hash-ground.txt' 
    analogy_list = functions.get_list_re(positive_set)
    non_analogy_list = functions.get_list_re(negative_set)
    samples = [(text, 'YES') for text in analogy_list] + [(text, 'NO') for text in non_analogy_list]
    train_data, train_labels, test_data, test_labels = functions.preprocess(samples, 0.5)
    overlap_input = [('nusvc','count'), ('svc', 'count')]
    prediction_second_input = []
    pipeline = []
    no_as_yes = [] # predictions with label NO classified with label YES
    yes_as_no = [] # predictions with label YES classified with label NO
    count = 0
    
    for element in overlap_input:
        pipeline = (Pipeline([(element[1], helpers.get_function(element[1])),
                      (element[0], helpers.get_function(element[0]))]))
        parameters = parameters_file.getparameters(element)
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, error_score=-1)
        grid_search.fit(train_data, train_labels)
        if count == 0:
            prediction = grid_search.best_estimator_.predict(test_data)
            matrix = confusion_matrix(test_labels, prediction, labels = ['YES', 'NO'])   
        else:
            prediction_second_input = grid_search.best_estimator_.predict(test_data)
            matrix = confusion_matrix(test_labels, prediction_second_input, labels = ['YES', 'NO'])   
        count += 1
        print(matrix)
        
    for i in range(len(test_labels)):
        if (test_labels[i] != prediction[i]) and (prediction[i] == prediction_second_input[i]):
            if test_labels[i] == 'NO':
                no_as_yes.append(test_data[i])
            else:
                yes_as_no.append(test_data[i])
    
    print("Overlapping NO as YES:")
    for i in range(len(no_as_yes)):
        print(no_as_yes[i])
    print("Overlapping YES as NO:")
    for i in range(len(yes_as_no)):
        print(yes_as_no[i])
    
main()
