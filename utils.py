import numpy as np
from sys import exit

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

def get_max_length(x_data):
    max_length = 0
    for x in x_data:
        if len(x) > max_length:
            max_length = len(x)
    return max_length