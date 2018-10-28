def add_rnn_opt(parser):
    parser.add_argument('-num_units', type=int, default=64,
                        help="rnn cell hidden size")

    parser.add_argument('-layer_num', type=int, default=1,
                        help="rnn cell hidden size")

    parser.add_argument('-cell_type', type=str, default='gru',
                        help="rnn cell hidden size")

    parser.add_argument('-bidirectional', action='store_true',
                        help='')

def add_nn_opt(parser):

    parser.add_argument('-model', type=str, default='Fasttext',
                        help="rnn cell hidden size")

    parser.add_argument('-embedding_size', type=int, default=128,
                        help="rnn cell hidden size")

    parser.add_argument('-vocab_size', type=int, default=5000,
                        help="rnn cell hidden size")

    parser.add_argument('-embedding_path', type=str, default=None,
                        help="rnn cell hidden size")

    parser.add_argument('-input_keep_prob', type=float, default=1.0,
                        help="rnn cell hidden size")

    parser.add_argument('-output_keep_prob', type=float, default=1.0,
                        help="rnn cell hidden size")

    parser.add_argument('-class_num', type=int, default=10,
                        help="rnn cell hidden size")

def add_train_opt(parser):
    parser.add_argument('-learning_rate', type=float, default=1e-3,
                        help="rnn cell hidden size")

    parser.add_argument('-batch_size', type=float, default=64,
                        help="rnn cell hidden size")

    parser.add_argument('-epoch_num', type=int, default=20,
                        help='')

    parser.add_argument('-print_every_step', type=int, default=100,
                        help='')

    parser.add_argument('-tensorboard_dir', type=str, default='tensorboard',
                        help='')

    parser.add_argument('-save_dir', type=str, default='save',
                        help='')



def add_data_opt(parser):
    parser.add_argument('-train_data', type=str, default='data/cnews/cnews.train.txt.seg',
                        help="rnn cell hidden size")

    parser.add_argument('-val_data', type=str, default='data/cnews/cnews.val.txt.seg',
                        help="rnn cell hidden size")

    parser.add_argument('-test_data', type=str, default='data/cnews/cnews.test.txt.seg',
                        help="rnn cell hidden size")

    parser.add_argument('-vocab_path', type=str, default='data/cnews/vocab.txt',
                        help="rnn cell hidden size")

    parser.add_argument('-label_path', type=str, default='data/cnews/label.txt',
                        help="rnn cell hidden size")

    parser.add_argument('-cut_length', type=int, default=600,
                        help="rnn cell hidden size")

    parser.add_argument('-reverse', action='store_true',
                        help="rnn cell hidden size")



