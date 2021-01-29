import numpy as np
from sys import exit
import pandas as pd

def list_2d_to_nparray(list_2d):
    new_array = []
    for line in list_2d:
        new_line = np.asarray(line)
        new_array.append(new_line)
    return np.asarray(new_array)

def encode_binary_labels(y):
    encoded_labels = []
    labels = []
    [labels.append(x) for x in y if x not in labels]
    print(labels)
    for label in y:
        if label == labels[0]:
            encoded_labels.append(0)
        elif label == labels[1]:
            encoded_labels.append(1)
        else:
            print(label)
            exit("wrong label input!")
    return encoded_labels

def get_npy_path(alertness, filename):
    return "training_data/"+alertness+"/"+filename+"_"+alertness+".npy"

def get_bin_path(alertness, name):
    return "training_data/"+alertness+"/"+name+"_"+alertness+".bin"

def get_bow_model_path(alertness):
    return "models/"+alertness+"_model/"+alertness+"_model_bow"

def get_bert_model_path(alertness):
    return "models/"+alertness+"_model/"+alertness+"_model_bert"

def get_max_length_path(alertness):
    return "models/"+alertness+"_model/"+alertness+"_model_bert/max_clause_len.txt"

def get_max_length(x_data):
    max_length = 0
    for x in x_data:
        if len(x) > max_length:
            max_length = len(x)
    return max_length


def max_length_padding(tokenised_clauses, max_length=0):
    """""""""""""""""""""""""""""""""""pad clauses with 0 to make them equal in length"""""""""""""""""""""""""""""""""""
    if max_length == 0:
        max_length = get_max_length(tokenised_clauses)
    else:
        max_length = max_length
    padded_clauses = []
    for clause in tokenised_clauses:
        padded_clause = clause
        while len(padded_clause) < max_length:
            padded_clause.append(0)
        padded_clauses.append(padded_clause)
    padded_clauses = list_2d_to_nparray(padded_clauses)
    print(padded_clauses.shape)

    return padded_clauses

def tokenise_clauses(clause, tokenizer):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(clause))

def read_predictions(predictions, mode):
    results = []
    standard_counter = 0
    worth_counter = 0
    if mode == "bow":
        for pd in predictions:
            if pd[0] > pd[1]:
                results.append("standard_trivial")
                standard_counter += 1
            elif pd[0] < pd[1]:
                results.append("worth_reading")
                worth_counter += 1
            else:
                exit("Wrong input suspected!")
        how_important = round(worth_counter / (standard_counter + worth_counter),4) * 100
        print("Ratio of worth reading clauses: "+str(how_important)+"%")
    elif mode == "bert":
        for pd in predictions:
            if pd[0] < 0.5:
                results.append("standard_trivial")
                standard_counter += 1
            elif pd[0] > 0.5:
                results.append("worth_reading")
                worth_counter += 1
            else:
                exit("Wrong input suspected!")
        how_important = round(worth_counter / (standard_counter + worth_counter), 4) * 100
        print("Ratio of worth reading clauses: " + str(how_important) + "%")
    else:
        exit("Wrong mode selection!")
    return np.array(results)

def store_results(document, results, filename, mode):
    classified_df = pd.DataFrame()
    classified_df['clause'] = document
    classified_df['class'] = results
    classified_df.to_csv("results/"+filename+'_classified_result_'+mode+'.csv')