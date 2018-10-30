import argparse
from text_classification.opt import *

def use_backend(backend='tensorflow'):
    if backend == 'tensorflow':
        from text_classification import main_tf as module

    else:
        from text_classification import main_torch as module

    return module



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_nn_opt(parser)
    add_rnn_opt(parser)
    add_cnn_opt(parser)
    add_train_opt(parser)
    add_data_opt(parser)
    opt = parser.parse_args()
    train = use_backend(opt.backend).train
    train(opt)
