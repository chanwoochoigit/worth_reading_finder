from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.models import load_model
import sys
from joblib import load
import numpy as np
import pandas as pd
from create_vocab import list_2d_to_nparray

"""helper function to convert clause to BOW vector"""

def vectorise_document(clauses, vocab):
    bow_vectors = []
    clause_counter = 0
    for clause in clauses:
        print("processing...{}/{}".format(clause_counter,len(clauses)))
        clause = clause.replace('.','').replace(',','').replace('eg','').replace('(','').replace(')','').split()
        temp = [0] * len(vocab)
        for i in range(len(vocab)):
            for word in clause:
                if vocab[i] in word:
                    temp[i] += 1
        bow_vectors.append(temp)
        clause_counter += 1
    return bow_vectors

def preprocess_document(document):
    """""""""""""""""""""standardise vectorised document"""""""""""""""""""""
    scaler = load('scaler.bin')
    document_scaled = scaler.fit_transform(document)
    # print(document_scaled)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""do feature selection on the vectorised test document"""""""""""
    fselector = load('fselector.bin')
    aa_scaled_selected = fselector.transform(document_scaled)
    test_document = aa_scaled_selected
    print(test_document.shape)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    return test_document

def read_predictions(predictions):
    results = []
    standard_counter = 0
    worth_counter = 0
    for pd in predictions:
        if pd[0] > pd[1]:
            results.append("standard_trivial")
            standard_counter += 1
        elif pd[0] < pd[1]:
            results.append("worth_reading")
            worth_counter += 1
        else:
            sys.exit("Wrong input suspected!")
    how_standard = round(standard_counter / (standard_counter + worth_counter),4) * 100
    print("This document is "+str(how_standard)+"% standard")
    return np.array(results)

def store_results(document, results):
    classified_df = pd.DataFrame()
    classified_df['clause'] = document
    classified_df['class'] = results
    classified_df.to_csv('classified_result.csv')

in_file = sys.argv[1]
print(in_file)

input_clauses = []

with open(in_file, 'r') as file:
    for line in file:
        if line != '' and line.isspace() is False:
            input_clauses.append(line)

"""""""""""""""""""""""""""""""""""""load x, vocab and model"""""""""""""""""""""""""""""""""""""
x = np.load('clause_vector.npy')
# y = encode_binary_labels(np.load('classes.npy', allow_pickle=True))
vocab = np.load('vocab.npy')
model = load_model('10000_12/best_model.hdf5')
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
clauses_vector = list_2d_to_nparray(vectorise_document(clauses=input_clauses, vocab=vocab))
# np.save('verizon.npy',clauses_vector)

"""""""""""""""""""""""load and preprocess vectorised document text file"""""""""""""""""""""""
# verizon = np.load('verizon.npy')
verizon_processed = preprocess_document(clauses_vector)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

predictions = model.predict(verizon_processed)
results = read_predictions(predictions)
print(results.shape)
store_results(input_clauses, results)