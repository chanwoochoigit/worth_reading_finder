import math
import bert
import sentencepiece
from numpy import random
from tensorflow.keras import layers
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import save_model, load_model

def tokenise_clauses(clause):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(clause))


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

"""""""""""""""""""""""""""""""""""load data and assign variables"""""""""""""""""""""""""""""""""""
data = pd.read_csv("data.csv")
clauses = data["clause"]
y = data["class"]
y = np.array(list(map(lambda x: 1 if x == "worth_reading" else 0, y)))
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""download and load bert tokeniser + vocabulary"""""""""""""""""""""""""""
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""hyper-parameters"""""""""""""
BATCH_SIZE = 32

VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 200
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = 2

DROPOUT_RATE = 0.2

NUM_EPOCHS = 10
""""""""""""""""""""""""""""""""""""""""""

#tokenise clauses
tokenised_clauses = [tokenise_clauses(clause) for clause in clauses]

"""""""""""""""get length of each clause to find the longest one to make all clauses to have equal length"""""""""""""""
clauses_with_len = [[clause, y[i], len(clause)] for i, clause in enumerate(tokenised_clauses)]
random.shuffle(clauses_with_len)
clauses_with_len.sort(key=lambda x: x[2])
sorted_clauses_labels = [(cls_label[0], cls_label[1]) for cls_label in clauses_with_len]
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""pad batches based on the batch size"""""""""""""""""""""""""""""""""""""""""
processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_clauses_labels, output_types=(tf.int32, tf.int32))
batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""divide training and test dataset"""""""""""""""""""""""""""""""""""""""""""
TOTAL_BATCHES = math.ceil(len(sorted_clauses_labels) / BATCH_SIZE)
TEST_BATCHES = TOTAL_BATCHES // 20
batched_dataset.shuffle(TOTAL_BATCHES)
test_data = batched_dataset.take(TEST_BATCHES)
training_data = batched_dataset.skip(TEST_BATCHES)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

classifier = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        cnn_filters=CNN_FILTERS,
                        dnn_units=DNN_UNITS,
                        model_output_classes=OUTPUT_CLASSES,
                        dropout_rate=DROPOUT_RATE)

""""""""""""""""""""""""""""compile classes; here we only use binary as there are 2 classes"""""""""""""""""""""""""""""
if OUTPUT_CLASSES == 2:
    classifier.compile(loss="binary_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])
else:
    classifier.compile(loss="sparse_categorical_crossentropy",
                       optimizer="adam",
                       metrics=["sparse_categorical_accuracy"])
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

classifier.fit(training_data, epochs=NUM_EPOCHS)

classifier.save('bert_model', save_format='tf')

results = classifier.evaluate(test_data)
print(results)
