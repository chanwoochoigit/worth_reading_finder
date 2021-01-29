import argparse
import sys

import bert
import sentencepiece
from numpy import random
from tensorflow.keras import layers
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import save_model, load_model
from utils import get_bert_model_path, tokenise_clauses, get_max_length_path, max_length_padding, read_predictions, \
    store_results

if __name__ == '__main__':

    # take flags
    parser = argparse.ArgumentParser()
    parser.add_argument("alertness", type=str)
    parser.add_argument("filename", type=str)
    args = parser.parse_args()

    # check flag validity
    valid_alertness = ["alice", "bob", "charlie"]
    alertness = args.alertness

    if alertness not in valid_alertness:
        sys.exit("Invalid argument!")

    filename = args.filename
    print(filename)

    input_clauses = []

    with open(filename, 'r') as file:
        for line in file:
            if line != '' and line.isspace() is False:
                input_clauses.append(line)

    """""""""""""""""""""""""""""""get maximum clause length for test data"""""""""""""""""""""""""""""""
    with open(get_max_length_path(alertness), "r") as max_len_file:
        max_len = int(max_len_file.read()[-3:])
    print(max_len)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""download and load bert tokeniser + vocabulary"""""""""""""""""""""""""""
    BertTokenizer = bert.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""""""""""""tokenise clauses with BERT tokeniser"""""""""""""""""""""""""""""""""""""""
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
    tokenised_clauses = [tokenise_clauses(clause, tokenizer) for clause in input_clauses]
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    #padd data
    padded_clauses = max_length_padding(tokenised_clauses, max_length=max_len)

    model = load_model(get_bert_model_path(alertness))
    predictions = model.predict(x=padded_clauses)
    print(predictions)
    results = read_predictions(predictions, "bert")
    print(results.shape)
    store_results(input_clauses, results, filename[:-4], "bert")


