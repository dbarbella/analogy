#By Davit Kvartskhava
''' This code includes a number of functions
	that can be used to explore the glove corpus and the multidimensional vector space associated with it.
	The main feature is the analogy finder (fill_the_gap method), which given three words, 
	A, B and C outputs some D. C to D is like A to B.'''

''' In order to run the fill_the_gap method with your set of words, just change the list_of_analogies global variable'''

'''The first thing that has to be run is parse_co_occurence_matrix, which parses the csv file 
	(one of the glove pre-trained vectors that can be downloaded here: https://github.com/stanfordnlp/GloVe) 
	and stores the vectors into a dictionary (<word, vector>) '''
import numpy

DICTIONARY = {}

RANGE = 20

#Returns file handle
def read_file(filename):
	try:
		handle = open(filename, 'r')
		return handle
	except:
		print("Enter the valid name for the filename!")

#parses co_occurence matrix and puts the <word, vector> pairs in global dictionaries.
def parse_co_occurence_matrix(filename):
	file = read_file(filename)

	for next_line in file:
		next_line_to_list = next_line.split() #split into tokens and add to array
		next_word = next_line_to_list[0] #the first token is the word itself 
		next_vec = numpy.array([float(i) for i in next_line_to_list[1:]]) #numpy arrays are efficient and with larger functionality.
		DICTIONARY[next_word] = next_vec

def print_item(X):
	X = X.lower()
	print("WORD: ", X, "\nVECTOR: ", DICTIONARY[X])

#returns the magnitude of a vector
def magnitude(vector):
	return numpy.linalg.norm(vector)

#assuming we are given two words in the string format,
#it returns the cosine value between the vectors associated
#with those words.
def cosine_for_words(word1, word2):
	vect1 = DICTIONARY[word1.lower()]
	vect2 = DICTIONARY[word2.lower()]
	return cosine(vect1, vect2)

#this is for vectors.
def cosine(vect1, vect2):

	magnitude_product = magnitude(vect1) * magnitude(vect2)
	if magnitude_product == 0: #we want to avoid division by 0.
		return 1
	else:
		return numpy.dot(vect1, vect2) / magnitude_product 

#given a base vector, returns a sorted array containing all the words
#that are sorted based on cosine similarity.
#The first element in the list will be contextually closest to the base vector.
def sort_by_cos_for_vector(base_vect):
	return_list = [] #list of tuples -> (cos_value, word)
	for next_word in DICTIONARY:

		next_cos_value = cosine(base_vect, DICTIONARY[next_word])
		return_list.append((next_cos_value, next_word))

	return_list.sort(key=lambda tup: tup[0], reverse = True)
	return return_list

#Same as the upper function, only difference is that the argument passed
#is a word instead of a vector. (don't judge me)
#returns an array sorted in a descending order by cos_value
def sort_by_cos_similarity(base_word):
	return_list = [] #list of tuples -> (cos_value, word)
	for next_word in DICTIONARY:

		next_cos_value = cosine_for_words(base_word, next_word)
		return_list.append((next_cos_value, next_word))

	return_list.sort(key=lambda tup: tup[0], reverse = True)
	return return_list

################################
#Given four words, prints the vectors AB and CD
def compare(A, B, C, D):

	first = DICTIONARY[A] - DICTIONARY[B]
	second = DICTIONARY[C] - DICTIONARY[D]
	print("subtract ", A, " from ", B, "\n Result: ",first)
	print("subtract ", C, " from ", D, "\n Result: ",second)

#works for descending order only
def find_closest_range(lst, pivot):
	result_list = []

	for i in range(len(lst)):#len(lst)
		next_word = lst[i]
		if pivot >= next_word[0]: #next_word[1] ---> cosine_for_words
			pivot_index = i
			break

	i = pivot_index #left pointer
	j = pivot_index #right pointer
	while(0 <= i and j < len(lst) and (j-i) <= RANGE):
		left_diff = abs(lst[i][0] - pivot)
		right_diff = abs(pivot - lst[j][0])

		if i == j:
			result_list.append(lst[i]) #WLOG
			i -= 1
			j += 1

		else:
			if left_diff < right_diff:
				result_list.append(lst[i])
				i -= 1
			else:
				result_list.append(lst[j])
				j += 1
	
	while(0 <= i and (j-i) <= RANGE):
		result_list.append(lst[i])
		i -= 1
	while(j < len(lst) and (j-i) <= RANGE):
		result_list.append(lst[j])
		j += 1


	return result_list

#Finding the cosine value between A and B. After that, 
#goes through the corpus and sorts every word by cosine value
#relative to the base_word. After that, finds the words that have 
#cosines with respect to base_word closer to that cosine between A and B.
#NOTE!!! ---- DOESN'T HAVE A GOOD PERFORMANCE
def fill_the_gap_cos(A, B, base_word):
	original_cos = cosine_for_word(A, B) 

	sorted_list = sort_by_cos_similarity(base_word)
	probable_ans_llist = find_closest_range(sorted_list, original_cos)
	
	for next_word in probable_ans_llist:
		print(next_word)

#find D =  B - A + C, and then
#find the words close to D based on cosine similarity. 
def fill_the_gap_euc(A, B, base_word):
	answer_vec = DICTIONARY[B] - DICTIONARY[A] + DICTIONARY[base_word]
	sorted_to_answer = sort_by_cos_for_vector(answer_vec)
	return sorted_to_answer[:RANGE]

#Merges both euclidean and cosine similarity to produce a 
#list of words that could be the possible correct answers. 
#In addition to that, we are not calculating cosine similarity
#from the origin, but from C.(makes a big difference)
def sort_by_cos_relative_to(direction_vect, relative_to_vect):#direction_vect ->> D, #relative_to_vect ->> C
	return_list_cos = [] #list of tuples -> (cos_value, word)
	return_list_dis = []
	return_list = []

	for next_word in DICTIONARY:

		next_cos_value = relative_cosine(direction_vect, DICTIONARY[next_word], relative_to_vect)#(B-A, next_vect, C)
		distance = magnitude(direction_vect - DICTIONARY[next_word])
		return_list_cos.append((next_cos_value, next_word))
		return_list_dis.append((distance, next_word))

	return_list_cos.sort(key=lambda tup: tup[0], reverse = True)
	return_list_dis.sort(key=lambda tup: tup[0], reverse = False)

	for i in range(100):
		next_dist_tuple = return_list_dis[i]
		for next_cos_tuple in return_list_cos:
			if next_cos_tuple[1] == next_dist_tuple[1]:
				return_list.append((next_cos_tuple[0], next_dist_tuple[1], next_dist_tuple[0]))
	return return_list
	

#relative_to_vect = C
def relative_cosine(vect1, vect2, relative_to_vect):
	new_vect1 = vect1 - relative_to_vect
	new_vect2 = vect2 - relative_to_vect

	return cosine(new_vect1, new_vect2)

#################################
'''Prints RANGE number of words A to B is like C to ____?'''
#################################
def fill_the_gap(A, B, base_word):
	relative_to_vect = DICTIONARY[base_word] #vector C
	direction_vect = DICTIONARY[B] - DICTIONARY[A] + relative_to_vect #Vector D
	sorted_to_answer = sort_by_cos_relative_to(direction_vect, relative_to_vect)

	for i in range(RANGE):
		print(sorted_to_answer[i])


parse_co_occurence_matrix("../glove.6B/glove.6B.100d.txt")

list_of_analogies = [("walk", "walked", "go"),("queen", "king", "woman"),("boy", "son", "girl"),("kitten", "cat", "puppy"),("france", "paris", "germany"), ("bird", "feather", "dog"), ("steam", "gas", "ice"), ("day", "night", "light"), ("dark", "darker", "windy"), ("like", "love", "dislike")]

for i in list_of_analogies:
	fill_the_gap(i[0], i[1], i[2])
	print("\n")
