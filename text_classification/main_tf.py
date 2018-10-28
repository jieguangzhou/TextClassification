import tensorflow as tf
import argparse
from .develop.timer import Timer
from .develop.IO import check_path
import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from text_classification.models_tf.fasttext import FastText
from text_classification.models_tf.textrnn import TextRNN
from text_classification.models_tf.textcnn import TextCNN
from text_classification.opt import *
from text_classification.data import batch_iter, load_data

message = 'step:{0:6}, train loss:{1:6.2}, train accuary:{2:7.2%}, val loss :{3:6.2}, val accuary:{4:7.2%}, cost:{5}'


def create_fasttext_model(vocab_size,
                          embedding_size,
                          class_num,
                          learning_rate,
                          keep_drop_prob=1.0,
                          embedding_path=None,
                          inference=False):
    if not inference:
        with tf.variable_scope('Fasttext', reuse=tf.AUTO_REUSE):
            train_model = FastText(vocab_size,
                                   embedding_size,
                                   class_num,
                                   learning_rate,
                                   keep_drop_prob,
                                   is_train=True,
                                   embedding_path=embedding_path)
    else:
        train_model = None
    with tf.variable_scope('Fasttext', reuse=tf.AUTO_REUSE):
        inference_model = FastText(vocab_size,
                                   embedding_size,
                                   class_num,
                                   learning_rate,
                                   keep_drop_prob,
                                   is_train=False,
                                   embedding_path=embedding_path)
    return train_model, inference_model


def create_textrnn_model(num_units,
                         layer_num,
                         vocab_size,
                         embedding_size,
                         class_num,
                         learning_rate,
                         keep_drop_prob=1.0,
                         cell_type='lstm',
                         bidirectional=False,
                         embedding_path=None,
                         inference=False):
    if not inference:
        with tf.variable_scope('TextRnn', reuse=tf.AUTO_REUSE):
            train_model = TextRNN(num_units,
                                  layer_num,
                                  vocab_size,
                                  embedding_size,
                                  class_num,
                                  learning_rate,
                                  keep_drop_prob,
                                  cell_type=cell_type,
                                  bidirectional=bidirectional,
                                  is_train=True,
                                  embedding_path=embedding_path)
    else:
        train_model = None

    with tf.variable_scope('TextRnn', reuse=tf.AUTO_REUSE):
        inference_model = TextRNN(num_units,
                                  layer_num,
                                  vocab_size,
                                  embedding_size,
                                  class_num,
                                  learning_rate,
                                  keep_drop_prob,
                                  cell_type=cell_type,
                                  bidirectional=bidirectional,
                                  is_train=False,
                                  embedding_path=embedding_path)

    return train_model, inference_model


def create_textcnn_model(filter_num,
                         kernel_sizes,
                         sequence_length,
                         vocab_size,
                         embedding_size,
                         class_num,
                         learning_rate,
                         keep_drop_prob=1.0,
                         embedding_path=None,
                         inference=False
                         ):
    if not inference:
        with tf.variable_scope('Fasttext', reuse=tf.AUTO_REUSE):
            train_model = TextCNN(filter_num,
                                  kernel_sizes,
                                  sequence_length,
                                  vocab_size,
                                  embedding_size,
                                  class_num,
                                  learning_rate,
                                  keep_drop_prob=keep_drop_prob,
                                  is_train=True,
                                  embedding_path=embedding_path)

    else:
        train_model = None

    with tf.variable_scope('Fasttext', reuse=tf.AUTO_REUSE):
        inference_model = TextCNN(filter_num,
                                  kernel_sizes,
                                  sequence_length,
                                  vocab_size,
                                  embedding_size,
                                  class_num,
                                  learning_rate,
                                  keep_drop_prob=keep_drop_prob,
                                  is_train=False,
                                  embedding_path=None)

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


def create_model(opt, inference=False):
    if opt.model.lower() == 'textrnn':
        train_model, inference_model = create_textrnn_model(num_units=opt.num_units,
                                                            layer_num=opt.layer_num,
                                                            vocab_size=opt.vocab_size,
                                                            embedding_size=opt.embedding_size,
                                                            class_num=opt.class_num,
                                                            learning_rate=opt.learning_rate,
                                                            keep_drop_prob=opt.keep_drop_prob,
                                                            cell_type=opt.cell_type,
                                                            bidirectional=opt.bidirectional,
                                                            embedding_path=opt.embedding_path,
                                                            inference=inference)
    elif opt.model.lower() == 'fasttext':
        train_model, inference_model = create_fasttext_model(opt.vocab_size,
                                                             opt.embedding_size,
                                                             opt.class_num,
                                                             opt.learning_rate,
                                                             keep_drop_prob=opt.keep_drop_prob,
                                                             embedding_path=opt.embedding_path,
                                                             inference=inference)

    elif opt.model.lower() == 'textcnn':
        train_model, inference_model = create_textcnn_model(opt.filter_num,
                                                            opt.kernel_sizes,
                                                            opt.cut_length,
                                                            opt.vocab_size,
                                                            opt.embedding_size,
                                                            opt.class_num,
                                                            opt.learning_rate,
                                                            opt.keep_drop_prob,
                                                            opt.embedding_path,
                                                            inference=inference)
    else:
        raise BaseException('没有这个模型')

    inference_model.print_parms()

    return train_model, inference_model


def train(opt):
    print('create model')
    train_model, inference_model = create_model(opt)
    save_path = os.path.join(opt.save_path)
    tensorboard_path = os.path.join(save_path, 'tensorborad')
    check_path(save_path, create=True)
    pickle.dump(opt, open(os.path.join(save_path, 'opt'), 'wb'))

    saver = tf.train.Saver(max_to_keep=1)

    # 读取train集和val集,并给予训练集合创建词典
    print('load data set')
    train_dataset = load_data(opt.train_data, opt.vocab_path, opt.label_path,
                              create_vocab=True, create_label_vocab=True, vocab_size=opt.vocab_size)
    val_dataset = load_data(opt.val_data, opt.vocab_path, opt.label_path)
    test_dataset = load_data(opt.test_data, opt.vocab_path, opt.label_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    timer = Timer()

    best_accuary = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(save_path)
        if ckpt:
            print('load model from : %s'%ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        summary_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
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
                    summary = tf.Summary(value=[
                        tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                        tf.Summary.Value(tag='train_accuary', simple_value=train_accuary),
                        tf.Summary.Value(tag='val_loss', simple_value=val_loss),
                        tf.Summary.Value(tag='val_accuary', simple_value=val_accuary),
                    ])
                    summary_writer.add_summary(summary, global_step)
                    cost_time = timer.cost_time()

                    print(message.format(global_step, train_loss, train_accuary, val_loss, val_accuary, cost_time))
                    if val_accuary > best_accuary:
                        best_accuary = val_accuary
                        saver.save(sess, os.path.join(save_path, inference_model.name), global_step=global_step)
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
    parser.add_argument()
    add_nn_opt(parser)
    add_rnn_opt(parser)
    add_train_opt(parser)
    add_cnn_opt(parser)
    opt = parser.parse_args()
    print(opt)
