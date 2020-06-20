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
from transformers import BertTokenizer, TFBertModel
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
from bert import tokenization

def convert_sentences_to_features(sess, sentences, tokenizer, max_seq_len=20):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    
    for sentence in sentences:
        input_ids, input_mask, segment_ids = convert_sentence_to_features(sentence, tokenizer, max_seq_len)
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)
    
    return all_input_ids, all_input_mask, all_segment_ids


def convert_sentence_to_features(sess, sentence, tokenizer, max_seq_len):
    tokens = ['[CLS]']
    tokens.extend(tokenizer.tokenize(sentence))
    if len(tokens) > max_seq_len-1:
        tokens = tokens[:max_seq_len-1]
    tokens.append('[SEP]')
    
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    #Zero Mask till seq_length
    zero_mask = [0] * (max_seq_len-len(tokens))
    input_ids.extend(zero_mask)
    input_mask.extend(zero_mask)
    segment_ids.extend(zero_mask)
    
    return input_ids, input_mask, segment_ids



def create_model(sess, args, vocab):
	#model = eval('network.classifier.CNN_Model')(args, vocab)
    #model = TFBertModel.from_pretrained('bert-base-uncased')
    
    bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    model = hub.Module(bert_path)
    tokenizer = tokenization.FullTokenizer(vocab_file='/data/yelp/train/train.txt', do_lower_case=True)

    input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
    input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None])
    segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])

    bert_inputs = dict(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids)

    bert_outputs = model(bert_inputs, signature="tokens", as_dict=True)

    import pdb; pdb.set_trace()

    sentences = ['New Delhi is the capital of India', 'The capital of India is Delhi']
    input_ids_vals, input_mask_vals, segment_ids_vals = convert_sentences_to_features(sentences, tokenizer, 20)

    out = sess.run(bert_outputs, feed_dict={input_ids: input_ids_vals, input_mask: input_mask_vals, segment_ids: segment_ids_vals})

    #out has two keys `dict_keys(['sequence_output', 'pooled_output'])`
    sentences = ['I prefer Python over Java', 'I like coding in Python', 'coding is fun']
    input_ids_vals, input_mask_vals, segment_ids_vals = convert_sentences_to_features(sentences, tokenizer, 20)

    out = sess.run(bert_outputs, feed_dict={input_ids: input_ids_vals, input_mask: input_mask_vals, segment_ids: segment_ids_vals})



    import pdb; pdb.set_trace()
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

    if not os.path.isfile(args.vocab):
        build_vocab(args.train_path, args.vocab)
    vocab = Vocabulary(args.vocab)
    print('vocabulary size', vocab.size)

    loader = ClassificationBatcher(args, vocab)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        model = create_model(sess, args, vocab)

        batches = loader.get_batches(mode='train')

        start_time = time.time()
        loss = 0.0
        best_dev = float('-inf')
        learning_rate = args.learning_rate

        for epoch in range(1, 1+args.max_epochs):
            print('--------------------epoch %d--------------------' % epoch)

            for batch in batches:
                results = model.run_train_step(sess, batch)
                step_loss = results['loss']
                loss += step_loss / args.train_checkpoint_step

                if results['global_step'] % args.train_checkpoint_step == 0:
                    print('iteration %d, time %.0fs, loss %.4f' \
                        % (results['global_step'], time.time() - start_time, loss))
                    loss = 0.0

                    val_batches = loader.get_batches(mode='valid')
                    acc, _, _ = model.run_eval(sess, val_batches)
                    print('valid accuracy %.4f' % acc)
                    if acc > best_dev:
                        best_dev = acc
                        print('Saving model...')
                        model.saver.save(sess, os.path.join(args.classifier_path, 'model'))

        test_batches = loader.get_batches(mode='test')
        acc, _, _ = model.run_eval(sess, test_batches)
        print('test accuracy %.4f' % acc)
