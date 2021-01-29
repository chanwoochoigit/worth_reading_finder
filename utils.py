import numpy as np

def list_2d_to_nparray(list_2d):
    new_array = []
    for line in list_2d:
        new_line = np.asarray(line)
        new_array.append(new_line)
    return np.asarray(new_array)

def get_npy_path(alertness, filename):
    return "training_data/"+alertness+"/"+filename+"_"+alertness+".npy"

def get_bin_path(alertness, name):
    return "training_data/"+alertness+"/"+name+"_"+alertness+".bin"

def get_bow_model_path(alertness):
    return "models/"+alertness+"_model/"+alertness+"_model_bow"

def get_bert_model_path(alertness):
    return "models/"+alertness+"_model/"+alertness+"_model_bert"
