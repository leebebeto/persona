import os
import sys
import time
import random

import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import network
from config import load_arguments
from vocab import Vocabulary, build_vocab
from dataloader.cnn_dataloader import ClassificationBatcher
from transformers import BertTokenizer, TFBertModel, TFBertForSequenceClassification
from tensorflow import keras
from tensorflow.keras import layers
#import tensorflow_hub as hub
#from bert import tokenization

def create_model(sess, args, vocab):
	#model = eval('network.classifier.CNN_Model')(args, vocab)
    #model = TFBertModel.from_pretrained('bert-base-uncased')
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')
#    input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
	

#    last_layer = tf.keras.layers.Dense(2)
#    softmax = tf.keras.layers.Activation('softmax')
#	import pdb; pdb.set_trace()
#	#final = softmax(last_layer(last_hidden_states))
	#input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
	#outputs = model(input_ids)
	#final_layer = keras.layers.Dense(2)(outputs)
    
    if args.load_model:
        print('Loading model from', os.path.join(args.classifier_path, 'model'))
        model.saver.restore(sess, os.path.join(args.classifier_path, 'model'))
    else:
        print('Creating model with fresh parameters.')
        sess.run(tf.global_variables_initializer())
    if not os.path.exists(args.classifier_path):
            os.makedirs(args.classifier_path)
    return model

if __name__ == '__main__':
    args = load_arguments()
    tf.executing_eagerly()
    tf.compat.v1.enable_eager_execution()


    if not os.path.isfile(args.vocab):
        build_vocab(args.train_path, args.vocab)
    vocab = Vocabulary(args.vocab)
    print('vocabulary size', vocab.size)

    loader = ClassificationBatcher(args, vocab)
    adam_optimizer = tf.keras.optimizers.Adam(0.0001, 0.5)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        model = create_model(sess, args, vocab)

        batches = loader.get_batches(mode='train')

        start_time = time.time()
        loss = 0.0
        best_dev = float('-inf')
        learning_rate = args.learning_rate
        bce = tf.keras.losses.BinaryCrossentropy()

        for epoch in range(1, 1+args.max_epochs):
            print('--------------------epoch %d--------------------' % epoch)

            for batch in batches:
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
                import pdb; pdb.set_trace()
                print_loss = tf.print(step_loss)
                sess.run(print_loss)
                print(step_loss)

                #step_loss = pred[0]
				#step_loss = results['loss']
#                loss += step_loss / args.train_checkpoint_step
#
#                if results['global_step'] % args.train_checkpoint_step == 0:
#                    print('iteration %d, time %.0fs, loss %.4f' \
		#                        % (results['global_step'], time.time() - start_time, loss))
#                    loss = 0.0
#
#                    val_batches = loader.get_batches(mode='valid')
#                    acc, _, _ = model.run_eval(sess, val_batches)
#                    print('valid accuracy %.4f' % acc)
#                    if acc > best_dev:
#                        best_dev = acc
#                        print('Saving model...')
#                        model.saver.save(sess, os.path.join(args.classifier_path, 'model'))
#
#        test_batches = loader.get_batches(mode='test')
#        acc, _, _ = model.run_eval(sess, test_batches)
#        print('test accuracy %.4f' % acc)
