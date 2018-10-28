import argparse
from text_classification.opt import *
from text_classification.main_tf import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_nn_opt(parser)
    add_rnn_opt(parser)
    add_cnn_opt(parser)
    add_train_opt(parser)
    add_data_opt(parser)
    opt = parser.parse_args()
    train(opt)
