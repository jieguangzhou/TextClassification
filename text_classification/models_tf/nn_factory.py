import tensorflow as tf
from tensorflow.contrib import rnn


def init_embedding(vocab_size, embedding_size, embedding_path=None, name='embedding'):
    embedding = tf.get_variable(name, [vocab_size, embedding_size])
    return embedding


def rnn_factory(num_units, layer_num, cell_type='lstm', input_keep_prob=1.0, output_keep_prob=1.0):
    if cell_type.lower() == 'lstm':
        cell_func = rnn.BasicLSTMCell
    elif cell_type.lower() == 'gru':
        cell_func = rnn.GRUCell

    else:
        cell_func = rnn.RNNCell

    cells = [cell_func(num_units) for _ in range(layer_num)]
    drop_func = lambda cell: rnn.DropoutWrapper(cell,
                                                input_keep_prob=input_keep_prob,
                                                output_keep_prob=output_keep_prob)
    cell = rnn.MultiRNNCell(list(map(drop_func, cells)))
    return cell
