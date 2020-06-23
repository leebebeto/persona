import os
import sys
import time
import random

import numpy as np

import tensorflow as tf
import network
from config import load_arguments
from vocab import Vocabulary, build_vocab
from dataloader.cnn_dataloader import ClassificationBatcher
from transformers import BertTokenizer, TFBertModel, TFBertForSequenceClassification
from tensorflow import keras
from tensorflow.keras import layers

if __name__ == '__main__':
    with tf.device('/device:gpu:5'):
        args = load_arguments()
        if not os.path.isfile(args.vocab):
            build_vocab(args.train_path, args.vocab)
        vocab = Vocabulary(args.vocab)
        print('vocabulary size', vocab.size)
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertModel.from_pretrained('bert-base-uncased')

        loader = ClassificationBatcher(args, vocab)
        adam_optimizer = tf.keras.optimizers.Adam(0.0001, 0.5)
    	#config = tf.ConfigProto()
    	#config.gpu_options.allow_growth = True

    		# model = create_model(sess, args, vocab)

        batches = loader.get_batches(mode='train')
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        start_time = time.time()
        loss = 0.0
        best_dev = float('-inf')
        learning_rate = args.learning_rate
        bce = tf.keras.losses.BinaryCrossentropy()
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        for epoch in range(1, 1+args.max_epochs):
            print('--------------------epoch %d--------------------' % epoch)

            for i, batch in enumerate(batches):
    #results = model.run_train_step(sess, batch)
                with tf.GradientTape(persistent=True) as tape:
                    labels = batch.labels
                    labels_list = [[i, abs(1-i)] for i in labels]
                    labels = tf.convert_to_tensor(labels_list)
                    output = model(batch.enc_batch)
                    last_layer = tf.keras.layers.Dense(2)
                    softmax = tf.keras.layers.Activation('softmax')
                    pred = softmax(last_layer(output[1]))
                    step_loss = bce(labels, pred)
    			
                grad_gen = tape.gradient(step_loss, model.trainable_variables)
                adam_optimizer.apply_gradients(zip(grad_gen, model.trainable_variables))
                print('step: ', i, 'loss: ', step_loss.numpy())








