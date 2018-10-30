import tensorflow as tf
from text_classification.models_tf.nn_factory import rnn_factory
from text_classification.models_tf.model import BinaryClassificationModel


class TextRNN(BinaryClassificationModel):
    """
    TextRNN
    """
    def __init__(self,
                 num_units,
                 layer_num,
                 vocab_size,
                 embedding_size,
                 class_num,
                 learning_rate,
                 keep_drop_prob=1.0,
                 cell_type='lstm',
                 bidirectional=False,
                 is_train=False,
                 embedding_path=None):

        self.num_units = num_units
        self.layer_num = layer_num
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.keep_drop_prob = keep_drop_prob if is_train else 1.0
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.is_train = is_train
        self.embedding_path = embedding_path
        super(TextRNN, self).__init__()

    def build(self):
        self._build_placeholder()
        self._build_embedding(self.vocab_size, self.embedding_size, self.embedding_path)
        input_sequences_emb = tf.nn.embedding_lookup(self.embedding, self.input_sequences)
        with tf.variable_scope('Rnn'):
            cell_fw = rnn_factory(self.num_units, self.layer_num, self.cell_type, output_keep_prob=self.keep_drop_prob)
            if self.bidirectional:
                cell_bw = rnn_factory(self.num_units, self.layer_num, self.cell_type, output_keep_prob=self.keep_drop_prob)
                rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                 cell_bw,
                                                                 input_sequences_emb,
                                                                 sequence_length=self.input_lengths,
                                                                 dtype=tf.float32)
                self.outputs = tf.concat(rnn_outputs, axis=2)

            else:
                self.outputs, _ = tf.nn.dynamic_rnn(cell_fw, input_sequences_emb, dtype=tf.float32)

            self.outputs_last_step = self.outputs[:, -1, :]

        self._build_output(self.outputs_last_step, class_num=self.class_num)
        if self.is_train:
            self._build_optimize(self.loss, self.learning_rate)
