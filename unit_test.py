from main_interface import *
from emails import send_email
import traceback
import unittest
import numpy as np

list_of_classifiers = ["naive", "svm", "max_ent", "neural"]
list_of_representation = ["count", "tfidf", "hash"]
list_of_extras = [{"sub_class":""}, {"sub_class":"linear", "stop_words":'english', "max_df":0.8, "activation":"tanh", "learning_rate":"adaptive"}]
positive_set = 'test_extractions/bc_samples.txt'
negative_set = 'test_extractions/bc_grounds.txt'
trial_info = 'Positive Analogy File: ' + positive_set + '\nNegative Analogy File: '+ negative_set + '\nExtra: ' +  '\n'

analogy_trial_output = [0.63076923076923075, np.array([[ 22, 142],[ 26, 265]]).tolist(), 0.13414634146341464, 0.45833333333333331, 0.20754716981132074]

class analogy_test(unittest.TestCase):
   
    def test_main_interface_output(self):
        self.assertEqual(analogy_trial(positive_set, negative_set, .5, "count", "naive", {"sub_class":""}, timer= 5), analogy_trial_output)
        
if __name__ == '__main__':
    unittest.main()
                                                                                                                                   
    #print(analogy_trial(positive_set, negative_set, .5, "count", "naive", {"sub_class":""}, timer= 5))
    
        

             

    
