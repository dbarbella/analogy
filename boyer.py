from py_algorithms.strings import new_boyer_moore_find
from boyer_moore import *
from nltk.corpus import gutenberg
from timeit import timeit, time

analogy_string_list = [['is', 'like'], ['are', 'like'], ['like', 'a'], ['like', 'an'], ['analogous'], ['analogy'], ['similar', 'to'], ['similarly'], ['the', 'same', 'as'], ['likewise'], ['equivalent', 'to'], ['identical', 'to'], ['comparable', 'to'], ['as', '', 'as'], ['was', 'like']]


paras = gutenberg.paras()
print("start")
start = time
para_indices = (find_any_patterns(paras, analogy_string_list))
end = time
print("manual is:" + end-start)
print("end")



# algorithm = new_boyer_moore_find()

# substr = 'abcd'
# a=algorithm('foo' + substr + 'bar', substr) # => 3

# print("manual is:" + end-start)

def find_boyer_moore(T, P):
    """ return lowest index of T at which the substring P begins or -1"""
    n, m = len(T), len(P)
    if m == 0: return 0
    last = {}               # Using hash table for fast access
    for k in range(m):
        last[P[k]] = k
    i = m - 1           # i index at T, k index at P
    k = m - 1                    # j index of last occurrence of T[i] in P
    while i < n:
        if T[i] == P[k]:            # if chars are equal
            if k == 0:
                return i     # check if Pattern is complete
            else:
                i -= 1                  # normal iteration
                k -= 1
        else:
            # if j < k (remember k index at P)
            # shift i += m - (j+1)
            # if j > k
            # shift i += m - k
            j = last.get(T[i], -1)     # -1 if item not there
            i += m - (min(k, j+1))
            k = m - 1
    return -1

def test_lists():
    print("Testing...")
    x = find_boyer_moore_lists(['in', 'the', 'stifling', 'heat', 'were', 'two', 'english', 'girls', 'in', 'traditional', 'blue', 'and', 'white', 'ghanaian', 'dress', 'playing', 'literary', 'bingo', 'and', 'demonstrating', 'solar', 'eclipses', 'with', 'a', 'football', 'and', 'an', 'orange.walking', 'back', 'with', 'laura', 'after', 'taking', 'pito', 'with', 'the', 'other', 'masters,', 'i', 'asked', 'her', 'what', 'it', 'was', 'like', 'living', 'and', 'working', 'in', 'ghana.'],['was', 'like'])
    print(x)

def find_boyer_moore_lists(T, P):
    """ return lowest index of T at which the sublist P begins or -1"""
    #print(T,P)
    n, m = len(T), len(P)
    if m == 0: return 0
    last = {}               # Using hash table for fast access
    for k in range(m):
        last[P[k]] = k
    i = m - 1           # i index at T, k index at P
    k = m - 1                    # j index of last occurrence of T[i] in P
    while i < n:
        if T[i] == P[k]:            # if chars are equal
            if k == 0:
                return i     # check if Pattern is complete
            else:
                i -= 1                  # normal iteration
                k -= 1
        else:
            # if j < k (remember k index at P)
            # shift i += m - (j+1)
            # if j > k
            # shift i += m - k
            j = last.get(T[i], -1)     # -1 if item not there
            i += m - (min(k, j+1))
            k = m - 1
    return -1

def find_boyer_moore_all(T, P):
    """ return all indices of T at which the substring P begins"""
    indices = []
    n, m = len(T), len(P)
    if m == 0: return 0
    last = {}               # Using hash table for fast access
    for k in range(m):
        last[P[k]] = k
    i = m - 1           # i index at T, k index at P
    k = m - 1                    # j index of last occurrence of T[i] in P
    while i < n:
        # print("Checking for i = " + str(i))
        if T[i].lower() == P[k] or P[k] == '':      # if things are equal, '' for as ... as case
            # print("Found a partial match at " + str(i) + ", with k = " + str(k))
            if k == 0:
                #print("Pattern complete.")
                indices += [i]     # check if Pattern is complete
                i += 1 # This might be wrong
            else:
                #print("Pattern not yet complete, updating i and k.")
                i -= 1                  # normal iteration
                k -= 1
                #print("i and k are now " + str(i) + " and " + str(k))
        else:
            # if j < k (remember k index at P)
            # shift i += m - (j+1)
            # if j > k
            # shift i += m - k
            # print("Not a match, moving forward.")
            j = last.get(T[i], -1)     # -1 if item not there
            i += m - (min(k, j+1))
            k = m - 1
    return indices


def find_boyer_moore_all_paras(paras_list, pattern):
    """
    return all indices of T at which the substring P begins
    T is now a list of paragraphs, each of which is a list of sentences
    boyer_moore is overkill here, probably
    """
    para_indices = []
    para_index = 0
    for para in paras_list:
        for sen in para:
            if find_boyer_moore_all(sen, pattern):
                para_indices += [para_index]
        para_index += 1
    return para_indices

# Very inefficient beta version - looks through the corpus once for each string.
def find_any_patterns(paras_list, pattern_list):
    para_indices = []
    for pattern in pattern_list:
        #print(pattern)
        next_indices = find_boyer_moore_all_paras(paras_list, pattern)
        #print(next_indices)
        para_indices = para_indices + next_indices
    return para_indices