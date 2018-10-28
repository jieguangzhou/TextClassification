import tensorflow as tf
import argparse
from develop.timer import Timer
from develop.IO import check_path
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from text_classification.models_tf.textrnn import TextRNN
from text_classification.models_tf.fasttext import FastText
from text_classification.opt import *
from text_classification.data import batch_iter, load_data

message = 'step:{0:6}, train loss:{1:6.2}, train accuary:{2:7.2%}, val loss :{3:6.2}, val accuary:{4:7.2%}, cost:{5}'


def create_textrnn_model(num_units,
                         layer_num,
                         vocab_size,
                         embedding_size,
                         class_num,
                         learning_rate,
                         input_keep_prob=1.0,
                         output_keep_prob=1.0,
                         cell_type='lstm',
                         bidirectional=False,
                         embedding_path=None):
    with tf.variable_scope('TextRnn', reuse=tf.AUTO_REUSE):
        train_model = TextRNN(num_units,
                              layer_num,
                              vocab_size,
                              embedding_size,
                              class_num,
                              learning_rate,
                              input_keep_prob,
                              output_keep_prob,
                              cell_type=cell_type,
                              bidirectional=bidirectional,
                              is_train=True,
                              embedding_path=embedding_path)
        train_model.print_parms()

    with tf.variable_scope('TextRnn', reuse=tf.AUTO_REUSE):
        inference_model = TextRNN(num_units,
                                  layer_num,
                                  vocab_size,
                                  embedding_size,
                                  class_num,
                                  learning_rate,
                                  input_keep_prob,
                                  output_keep_prob,
                                  cell_type=cell_type,
                                  bidirectional=bidirectional,
                                  is_train=False,
                                  embedding_path=embedding_path)

    return train_model, inference_model


def create_fasttext_model(vocab_size,
                          embedding_size,
                          class_num,
                          learning_rate,
                          embedding_path=None):
    with tf.variable_scope('Fasttext', reuse=tf.AUTO_REUSE):
        train_model = FastText(vocab_size,
                               embedding_size,
                               class_num,
                               learning_rate,
                               is_train=True,
                               embedding_path=embedding_path)
        train_model.print_parms()

    with tf.variable_scope('Fasttext', reuse=tf.AUTO_REUSE):
        inference_model = FastText(vocab_size,
                               embedding_size,
                               class_num,
                               learning_rate,
                               is_train=False,
                               embedding_path=embedding_path)
    return train_model, inference_model


def get_feed_dict(model, sequences, labels, lengths):
    feed_dict = {model.input_sequences: sequences,
                 model.input_labels: labels,
                 model.input_lengths: lengths}
    return feed_dict


def evaluate(sess, model, dataset, batch_size=64, reverse=False, cut_length=None):
    """评估模型在特定数据集上的loss和accuracy"""
    total_num = len(dataset[0])
    total_loss = 0
    total_accuracy = 0
    for sequences, labels, lengths in batch_iter(*dataset,
                                                 batch_size=batch_size,
                                                 reverse=reverse,
                                                 cut_length=cut_length):
        loss, accuracy = sess.run([model.loss, model.accuracy],
                                  feed_dict=get_feed_dict(model, sequences, labels, lengths))
        batch_num = len(labels)
        total_loss += batch_num * loss
        total_accuracy += batch_num * accuracy

    return total_loss / total_num, total_accuracy / total_num


def create_model(opt):
    if opt.model.lower() == 'textrnn':
        train_model, inference_model = create_textrnn_model(num_units=opt.num_units,
                                                            layer_num=opt.layer_num,
                                                            vocab_size=opt.vocab_size,
                                                            embedding_size=opt.embedding_size,
                                                            class_num=opt.class_num,
                                                            learning_rate=opt.learning_rate,
                                                            input_keep_prob=opt.input_keep_prob,
                                                            output_keep_prob=opt.output_keep_prob,
                                                            cell_type=opt.cell_type,
                                                            bidirectional=opt.bidirectional,
                                                            embedding_path=opt.embedding_path)
    elif opt.model.lower() == 'fasttext':
        train_model, inference_model = create_fasttext_model(opt.vocab_size,
                                                             opt.embedding_size,
                                                             opt.class_num,
                                                             opt.learning_rate,
                                                             embedding_path=None)
    else:
        raise BaseException('没有这个模型')

    return train_model, inference_model


def train(opt):
    train_model, inference_model = create_model(opt)

    # 读取train集和val集,并给予训练集合创建词典
    train_dataset = load_data(opt.train_data, opt.vocab_path, opt.label_path,
                              create_vocab=True, create_label_vocab=True, vocab_size=opt.vocab_size)
    val_dataset = load_data(opt.val_data, opt.vocab_path, opt.label_path)
    test_dataset = load_data(opt.test_data, opt.vocab_path, opt.label_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    timer = Timer()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(opt.epoch_num):
            epoch_key = 'Epoch: %s' % (epoch + 1)
            print(epoch_key)
            timer.mark(epoch_key)
            total_loss, total_accuracy, total_num = 0, 0, 0
            train_batch_data = batch_iter(*train_dataset,
                                          batch_size=opt.batch_size,
                                          reverse=opt.reverse,
                                          cut_length=opt.cut_length)
            for sequences, labels, lengths in train_batch_data:
                loss, accuracy, global_step, _ = sess.run(
                    [train_model.loss, train_model.accuracy, train_model.global_step, train_model.optimize],
                    feed_dict=get_feed_dict(train_model, sequences, labels, lengths))

                batch_num = len(labels)
                total_num += batch_num
                total_loss += batch_num * loss
                total_accuracy += batch_num * accuracy

                if global_step % opt.print_every_step == 0:
                    train_loss = total_loss / total_num
                    train_accuary = total_accuracy / total_num

                    val_loss, val_accuary = evaluate(sess, inference_model, val_dataset,
                                                     batch_size=opt.batch_size,
                                                     reverse=opt.reverse,
                                                     cut_length=opt.cut_length)
                    cost_time = timer.cost_time()

                    print(message.format(global_step, train_loss, train_accuary, val_loss, val_accuary, cost_time))
                    total_loss, total_accuracy, total_num = 0, 0, 0

        test_loss, test_accuary = evaluate(sess, inference_model, test_dataset,
                                           batch_size=opt.batch_size,
                                           reverse=opt.reverse,
                                           cut_length=opt.cut_length)
        cost_time = timer.cost_time()
        print('eval test data')
        print('loss:{0:6.2}, accuary:{1:7.2%}, cost:{2}'.format(test_loss, test_accuary, cost_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_nn_opt(parser)
    add_rnn_opt(parser)
    add_train_opt(parser)
