import os
# import tensorflow as tf
import tensorflow.compat.v1 as tf

from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')
input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
outputs = model(input_ids)
last_hidden_states = outputs[0]






