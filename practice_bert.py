import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, TFBertForSequenceClassification
from tensorflow import keras
from tensorflow.keras import layers

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')
input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

last_layer = tf.keras.layers.Dense(2)
softmax = tf.keras.layers.Activation('softmax')
final = softmax(last_layer(last_hidden_states))


import pdb; pdb.set_trace()













