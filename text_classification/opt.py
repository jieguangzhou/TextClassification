def add_rnn_opt(parser):
    group = parser.add_argument_group('rnn')

    group.add_argument('-num_units', type=int, default=64,
                        help="rnn cell hidden size")

    group.add_argument('-layer_num', type=int, default=1,
                        help="rnn layer number")

    group.add_argument('-cell_type', type=str, default='gru',
                        help="rnn cell type, gru or lstm")

    group.add_argument('-bidirectional', action='store_true',
                        help='use bidirectional')

def add_cnn_opt(parser):
    group = parser.add_argument_group('cnn')

    group.add_argument('-filter_num', type=int, default=128,
                        help="cnn filter num")

    group.add_argument('-kernel_sizes', type=int, nargs='+', default=[5],
                        help="cnn kernel_sizes, a list of int")


def add_nn_opt(parser):
    group = parser.add_argument_group('nn')

    group.add_argument('-model', type=str, default='Fasttext',
                        help="use model, fasttext, textrnn or textcnn")

    group.add_argument('-embedding_size', type=int, default=128,
                        help="embedding size")

    group.add_argument('-vocab_size', type=int, default=5000,
                        help="vocab size")

    group.add_argument('-embedding_path', type=str, default=None,
                        help="embedding path, 暂不使用")

    group.add_argument('-keep_drop_prob', type=float, default=0.5,
                        help="keep_drop_prob")

    group.add_argument('-class_num', type=int, default=10,
                        help="class_num")

def add_train_opt(parser):
    group = parser.add_argument_group('train')

    group.add_argument('-learning_rate', type=float, default=1e-3,
                        help="learning_rate")

    group.add_argument('-batch_size', type=float, default=64,
                        help="batch_size")

    group.add_argument('-epoch_num', type=int, default=10)

    group.add_argument('-print_every_step', type=int, default=100)

    group.add_argument('-save_path', type=str, default='save')

    group.add_argument('-backend', type=str, default='tensorflow',
                       help='tensorflow or torch')


def add_server_opt(parser):
    group = parser.add_argument_group('server')

    group.add_argument('-port', type=int, default=9999,
                        help="端口号")


def add_data_opt(parser):
    group = parser.add_argument_group('data')
    group.add_argument('-train_data', type=str, default='data/cnews/cnews.train.txt.seg',
                        help="train data path")

    group.add_argument('-val_data', type=str, default='data/cnews/cnews.val.txt.seg',
                        help="val data path")

    group.add_argument('-test_data', type=str, default='data/cnews/cnews.test.txt.seg',
                        help="test data path")

    group.add_argument('-vocab_path', type=str, default='data/cnews/vocab.txt',
                        help="vocab_pathe")

    group.add_argument('-label_path', type=str, default='data/cnews/label.txt',
                        help="label_path")

    group.add_argument('-cut_length', type=int, default=600,
                        help="cut_length")

    group.add_argument('-reverse', action='store_true',
                        help="reverse the sequence")



