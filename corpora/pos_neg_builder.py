'''
pos_neg_builder.py

This file is responsible for taking a set of corpora, labeled in one of several ways,
and producing a set out output files - for now, one containing the positive set, and one containing the
negative set.

The two input formats supported are:

name,sentence-text,structural-score,attributional-score,validity

This is called "two_factor"

and

name,sentence-text,analogy-score,validity

This is called "one_factor"

'''

import csv

ONE_FACTOR_SENTENCE_COLUMN = 1
ONE_FACTOR_SCALAR_COLUMN = 2
ONE_FACTOR_VALID_COLUMN = 3

TWO_FACTOR_SENTENCE_COLUMN = 1
TWO_FACTOR_STR_COLUMN = 2
TWO_FACTOR_ATT_COLUMN = 3
TWO_FACTOR_VALID_COLUMN = 4


def one_factor_cutoff(scalar):
    """
    :param scalar: A value from 1 to 3
    :return: True if it's an analogy, false otherwise
    """
    if 1 <= scalar <= 3:
        return scalar > 2
    else:
        raise ValueError("In one_factor_cutoff, scalar was out of bounds.")


def two_factor_cutoff(structural, attributional):
    """
    :param structural: The structural score, from 1 to 3
    :param attributional: The attributional score, from 1 to 3
    :return:
    """
    if 1 <= structural <= 3 and 1 <= attributional <= 3:
        print(structural, attributional)
        print(structural > attributional)
        return structural > attributional
    else:
        raise ValueError("In two_factor_cutoff, one of the values was out of bounds.")


def validity_check(validity_val):
    if validity_val == 'F':
        return False
    else:
        return True

def produce_pos_neg_files(dict_of_input_files, pos_file_name, neg_file_name,
                          one_factor_func=one_factor_cutoff, two_factor_func=two_factor_cutoff,
                          in_delimiter=','):
    """
    :param dict_of_input_files: This is a dictionary with file names as keys and "one_factor" or "two_factor"
    as values.
    :param pos_file_name: The file to which we want to write the positive examples.
    :param neg_file_name: The file to which we want to write the negative examples.
    :param one_factor_func: The function that determines whether a one-factor example is an analogy.
    :param two_factor_func: The function that determines whether a two-factor example is an analogy.
    :param in_delimiter: The delimiter used in the input files. A single comma by default.
    :return: None
    """
    with open(pos_file_name, 'w', newline='', encoding="utf-8") as pos_file, \
            open(neg_file_name, 'w', newline='', encoding="utf-8") as neg_file:
        pos_writer = csv.writer(pos_file, quoting=csv.QUOTE_ALL)
        neg_writer = csv.writer(neg_file, quoting=csv.QUOTE_ALL)
        for file_name in dict_of_input_files.keys():
            input_type = dict_of_input_files[file_name]
            with open(file_name, 'r', encoding="utf-8") as file:
                csv_reader = csv.reader(file, delimiter=in_delimiter)  # How can we indicate a header row?

                csv_reader.__next__()  # Move past the header row; this is not a great solution.

                if input_type == "one_factor":
                    for row in csv_reader:  # Consider refactoring this so that only one var holds the func name.
                        sentence = row[ONE_FACTOR_SENTENCE_COLUMN]
                        scalar = float(row[ONE_FACTOR_SCALAR_COLUMN])
                        valid = validity_check(row[ONE_FACTOR_VALID_COLUMN])
                        if valid:
                            is_analogy = one_factor_func(scalar)
                            if is_analogy:
                                # We would like this to drop the entire row into pos_file_name
                                pos_writer.writerow(row)
                            else:
                                # We would like this to drop the entire row into neg_file_name
                                neg_writer.writerow(row)

                elif input_type == "two_factor":
                    # We need a smarter way to handle header rows.
                    for row in csv_reader:  # Consider refactoring this so that only one var holds the func name.
                        sentence = row[TWO_FACTOR_SENTENCE_COLUMN]
                        structural = float(row[TWO_FACTOR_STR_COLUMN])
                        attributional = float(row[TWO_FACTOR_ATT_COLUMN])
                        valid = validity_check(row[TWO_FACTOR_VALID_COLUMN])
                        if valid:
                            is_analogy = two_factor_func(structural, attributional)
                            if is_analogy:
                                # We would like this to drop the entire row into pos_file_name
                                print("Writing pos", row)
                                pos_writer.writerow(row)
                            else:
                                # We would like this to drop the entire row into neg_file_name
                                print("Writing neg", row)
                                neg_writer.writerow(row)

'''
def exportCSV(fileName, samples):
    with open(fileName, 'w') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(['id', 'text', 'ratings'])

        for key, value in samples.items():
            text = value[0]
            ratingList = [rating for rating in value[1]]
            outputRow = [key, text]
            outputRow.extend(ratingList)
            writer.writerow(outputRow)
        print(fileName + ".csv export sucess.")


def readCSV(file_name, sentence_column):
    """
    :param file_name: The file name of the CSV to read
    :param sentence_column: The column within the CSV that contains the sentences.
    Try to make sure this is 1, for consistency.
    :return: A list of the sentences, as strings.
    """
    sent = []
    with open(file_name) as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            sentence = row[sentence_column]
            sent.append(sentence)
    return sent
'''

if __name__ == '__main__':
    file_dict = {#"./oanc/analogy_names_OANC-TRAV-label1.tsv": "two_factor",
                 "./oanc/analogy_names_OANC-NONFIC-label1.tsv": "two_factor"}
    produce_pos_neg_files(file_dict,
                          "nonfic-pos-test-file.csv",
                          "nonfic-neg-test-file.csv",
                          in_delimiter='\t')