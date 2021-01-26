import bert
import sentencepiece
from numpy import random
from tensorflow.keras import layers
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import save_model, load_model

classifier = load_model('bert_model')