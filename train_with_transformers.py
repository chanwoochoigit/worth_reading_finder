import argparse
import math
import sys

import bert
import sentencepiece
from numpy import random
from transformers import BertTokenizer
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import save_model, load_model
from utils import get_bert_model_path, get_max_length, max_length_padding, tokenise_clauses, add_special_tokens
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from transformers import TFBertForSequenceClassification

def formatise_bert_input(clauses, labels):
    tokeniser = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    bert_input = []
    max_length = get_max_length(clauses)
    token_type_ids = [0] * max_length   #token types are zero because we are doing text classification not QA

    """""""""instances"""""""""
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []

    for i in range(len(clauses)):
        print("processing...{}/{}".format(i,len(clauses)))

        # create input ids
        cls_tokenised = tokeniser.tokenize(clauses[i])
        input_ids = tokeniser.convert_tokens_to_ids(cls_tokenised)
        input_ids_padded = input_ids + ([0] * (max_length - len(input_ids)))

        #create attention mask
        attention_mask = [1] * len(input_ids)   #give more focus on non-padded tokens
        attention_mask = attention_mask + ([0] * (max_length - len(input_ids)))

        #update each list
        input_ids_list.append(input_ids_padded)
        token_type_ids_list.append(token_type_ids)
        attention_mask_list.append(attention_mask)
        label_list.append(labels[i])

    return tf.data.Dataset.from_tensor_slices((input_ids_list, token_type_ids_list, attention_mask_list, label_list)).map(map_to_dict)

"""""""""""""""helper function to map bert inputs to dictionary format"""""""""""""""
def map_to_dict(token_ids, token_type_ids, attention_mask, label):
    return {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            }, label
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class TEXT_MODEL(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)
        l_2 = self.cnn_layer2(l)
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)

        concatenated = tf.concat([l_1, l_2, l_3], axis=-1)  # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output


if __name__ == '__main__':

    # take flags
    parser = argparse.ArgumentParser()
    parser.add_argument("alertness", type=str)
    args = parser.parse_args()

    # check flag validity
    valid_alertness = ["alice", "bob", "charlie"]
    alertness = args.alertness

    if alertness not in valid_alertness:
        sys.exit("Invalid argument!")

    """""""""""""""""""""""""""""""""""load data and assign variables"""""""""""""""""""""""""""""""""""
    data = pd.read_csv("training_data/"+alertness+"/data_"+alertness+".csv")
    clauses = data["clause"]
    clauses_with_special_tokens = add_special_tokens(clauses)
    y = data["class"]
    y = np.array(list(map(lambda x: 1 if x == "worth_reading" else 0, y)))
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    print(len(clauses_with_special_tokens) // 5 * 4)
    print(len(y))
    X_train, X_test, y_train, y_test = train_test_split(clauses_with_special_tokens, y, test_size=0.2)

    """""""""""""""""""""""""""tokenise clauses with pretrained BERT tokeniser"""""""""""""""""""""""""""
    bert_input_train = formatise_bert_input(X_train, y_train)
    bert_input_test = formatise_bert_input(X_test, y_test)

    print(bert_input_train)
    print(bert_input_test)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""""""set hyper-parameters and compile the model"""""""""""""""""""""""""""""""""
    num_epochs = 3
    learning_rate = 2e-5

    #init model
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
    loss = tf.keras.losses.binary_crossentropy(from_logits=True)
    metric = tf.keras.metrics.binary_accuracy('accuracy')
    model.compile(optimizer=optimiser, loss=loss, metrics=[metric])
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""""train model using the bert tokenised dataset"""""""""""""""""""""""""""""""
    bert_history = model.fit(bert_input_train, epochs=num_epochs, validation_data=bert_input_test)

    #
    # classifier = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
    #                         embedding_dimensions=EMB_DIM,
    #                         cnn_filters=CNN_FILTERS,
    #                         dnn_units=DNN_UNITS,
    #                         model_output_classes=OUTPUT_CLASSES,
    #                         dropout_rate=DROPOUT_RATE)
    #
    # """"""""""""""""""""""""""""compile classes; here we only use binary as there are 2 classes"""""""""""""""""""""""""""""
    # if OUTPUT_CLASSES == 2:
    #     classifier.compile(loss="binary_crossentropy",
    #                        optimizer="adam",
    #                        metrics=["accuracy"])
    # else:
    #     classifier.compile(loss="sparse_categorical_crossentropy",
    #                        optimizer="adam",
    #                        metrics=["sparse_categorical_accuracy"])
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #
    # """"""""""""""""""""""""""""""""""""""""divide training & test set and train"""""""""""""""""""""""""""""""""""""""""
    # X_train, X_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)
    #
    # classifier.fit(x=X_train, y=y_train, epochs=NUM_EPOCHS)
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #
    # """"""""""""""""""""""""""""""""""""""""""""""save model and max length"""""""""""""""""""""""""""""""""""""""""""""
    # classifier.save(get_bert_model_path(alertness), save_format='tf')
    # with open(get_bert_model_path(alertness)+"/max_clause_len.txt", "w") as text_file:
    #     text_file.write("max_length: "+str(max_length))
    #
    # results = classifier.evaluate(x=X_test, y=y_test)
    # print(results)
