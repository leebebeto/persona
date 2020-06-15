import os
import sys
import time
import random
import logging

import numpy as np
import pandas as pd
# import tensorflow as tf
import tensorflow.compat.v1 as tf
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

from tensorflow.python.client import device_lib

import network
from utils import *
from vocab import Vocabulary, build_unify_vocab
from config import load_arguments
from dataloader.multi_style_dataloader import MultiStyleDataloader
from dataloader.online_dataloader import OnlineDataloader

smoothie = SmoothingFunction().method4

logger = logging.getLogger(__name__)

def write_origin_file(corpus, f_name):
    file = open(f_name, 'w')
    vstr = ''
    
    for sentence in corpus:
        vstr = vstr + str(sentence)
        vstr += '\n'
        
    file.writelines(vstr)
    file.close()
    print('file save: ', f_name)
    
def write_transfer_file(corpus, f_name):
    file = open(f_name, 'w')
    vstr = ''
    
    for sentence in corpus:
        for word in sentence :
            vstr = vstr + str(word) + ' '
        vstr += '\n'
        
    file.writelines(vstr)
    file.close()
    print('file save: ', f_name)
    
def write_csv_file_with_label(label, pred_label, origin, transfer):
    process_ref = []
    origin_sentence = []
    transfer_sentence = []
    
    for sentence in origin:
        origin_sentence.append(sentence)
    
    for sentence in transfer:
        vstr = ''
        for word in sentence:
            vstr = vstr + str(word) + ' '
        transfer_sentence.append(vstr)
    
    df = pd.DataFrame({'label': label, 'pred_label': pred_label, 'origin': origin_sentence, 'transfer': transfer_sentence})
    df.to_csv('transferred_result.csv', index=False, encoding='UTF8')
    

#print(device_lib.list_local_devices())
def evaluation(sess, args, batches, model, 
    classifier, classifier_vocab, domain_classifer, domain_vocab,
    output_path, write_dict, save_samples=False, mode='valid', domain=''):
    
#     origin = origin sentence
#     label = label for transferring
#     pred_label = predicted label
#     transfer = transferred sentence
#     hypo = transferred sentence but using BLEU score (same as transfer)
    
    
    transfer_acc = 0
    domain_acc = 0
    origin_acc = 0
    total = 0
    domain_total =0
    ref = []
    ori_ref = []
    hypo = []
    origin = []
    transfer = []
    label = []
    pred_label = []
    reconstruction = []
    accumulator = Accumulator(len(batches), model.get_output_names(domain))

    for batch in batches:
        results = model.run_eval_step(sess, batch, domain)
        accumulator.add([results[name] for name in accumulator.names])

        rec = [[domain_vocab.id2word(i) for i in sent] for sent in results['rec_ids']]
        rec, _ = strip_eos(rec)

        tsf = [[domain_vocab.id2word(i) for i in sent] for sent in results['tsf_ids']]
        tsf, lengths = strip_eos(tsf)

        reconstruction.extend(rec)
        transfer.extend(tsf)
        hypo.extend(tsf)
        label.extend(batch.labels)
        origin.extend(batch.original_reviews)
        for x in batch.original_reviews:
            ori_ref.append([x.split()])
        for x in batch.references:
            ref.append([x.split()])

        # tansfer the output sents into classifer ids for evaluation
        tsf_ids = batch_text_to_ids(tsf, classifier_vocab)
        # evaluate acc
        feed_dict = {classifier.input: tsf_ids,
                     classifier.enc_lens: lengths,
                     classifier.dropout: 1.0}
        preds = sess.run(classifier.preds, feed_dict=feed_dict)
        pred_label.extend(preds)
        trans_label = batch.labels == 0
        transfer_acc += np.sum(trans_label == preds)
        total += len(trans_label)

        # evaluate domain acc
        if domain == 'target':
            domian_ids = batch_text_to_ids(tsf, domain_vocab)
            feed_dict = {domain_classifier.input: domian_ids,
                         domain_classifier.enc_lens: lengths,
                         domain_classifier.dropout: 1.0}
            preds = sess.run(domain_classifier.preds, feed_dict=feed_dict)
            domain_acc += np.sum(preds == 1)
            domain_total += len(preds)
            
    write_origin_file(origin, 'yahoo_lyrics_origin.txt')
    write_transfer_file(transfer, 'yahoo_lyrics_transfer.txt')
    write_csv_file_with_label(label, pred_label, origin, transfer)


def create_model(sess, args, vocab):
    model = eval('network.' + args.network + '.Model')(args, vocab)
    if args.load_model:
        logger.info('-----Loading styler model from: %s.-----' % os.path.join(args.styler_path, 'model'))
        model.saver.restore(sess, os.path.join(args.styler_path, 'model'))
    else:
        logger.info('-----Creating styler model with fresh parameters.-----')
        sess.run(tf.global_variables_initializer())
    if not os.path.exists(args.styler_path):
            os.makedirs(args.styler_path)
    return model

# elimiate the first variable scope, and restore the classifier from the path
def restore_classifier_by_path(classifier, classifier_path, scope):
    new_vars = {}
    for var in classifier.params:
        pos = var.name.find('/')
        # eliminate the first variable scope, e.g., target, source
        new_vars[var.name[pos+1:-2]] = var
    saver = tf.train.Saver(new_vars)
    saver.restore(sess, os.path.join(classifier_path, 'model'))
    logger.info("-----%s classifier model loading from %s successfully!-----" % (scope, classifier_path))

if __name__ == '__main__':
    args = load_arguments()
    assert args.domain_adapt, "domain_adapt arg should be True."

    if not os.path.isfile(args.multi_vocab):
        build_unify_vocab([args.target_train_path, args.source_train_path], args.multi_vocab)
    multi_vocab = Vocabulary(args.multi_vocab)
    logger.info('vocabulary size: %d' % multi_vocab.size)

    # use tensorboard
    if args.suffix:
        tensorboard_dir = os.path.join(args.logDir, 'tensorboard', args.suffix)
    else:
        tensorboard_dir = os.path.join(args.logDir, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    write_dict = {
    'writer': tf.summary.FileWriter(logdir=tensorboard_dir, filename_suffix=args.suffix),
    'step': 0
    }

    # load data
    loader = MultiStyleDataloader(args, multi_vocab)
    # create a folder for data samples
    source_output_path = os.path.join(args.logDir, 'domain_adapt', 'source')
    if not os.path.exists(source_output_path):
        os.makedirs(source_output_path)
    target_output_path = os.path.join(args.logDir, 'domain_adapt', 'target')
    if not os.path.exists(target_output_path):
        os.makedirs(target_output_path)

    # whether use online dataset for testing
    if args.online_test:
        online_data = OnlineDataloader(args, multi_vocab)
        online_data = online_data.online_test
        output_online_path = os.path.join(args.logDir, 'domain_adapt', 'online-test')
        if not os.path.exists(output_online_path):
            os.mkdir(output_online_path)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # create style transfer model
        model = create_model(sess, args, multi_vocab)

        # vocabulary for classifer evalution
        with tf.variable_scope('target'):
            target_vocab = Vocabulary(args.target_vocab)
            target_classifier = eval('network.classifier.CNN_Model')(args, target_vocab, 'target')
            restore_classifier_by_path(target_classifier, args.target_classifier_path, 'target')

        with tf.variable_scope('domain'):
            domain_classifier = eval('network.classifier.CNN_Model')(args, multi_vocab, 'domain')
            restore_classifier_by_path(domain_classifier, args.domain_classifier_path, 'domain')

        # load training dataset
        source_batches = loader.get_batches(domain='source', mode='train')
        target_batches = loader.get_batches(domain='target', mode='train')

        # testing
        test_batches = loader.get_batches(domain='target', mode='test')
        logger.info('---testing target domain:')
        evaluation(sess, args, test_batches, model, 
            target_classifier, target_vocab, domain_classifier, multi_vocab,
            os.path.join(target_output_path, 'test'), write_dict, mode='test', domain='target')
