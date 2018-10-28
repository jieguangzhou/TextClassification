import tensorflow as tf
from tensorflow.contrib import rnn
from text_classification.nn_factory import init_embedding, rnn_factory
from text_classification.models_tf.model import BinaryClassificationModel


class TextRNN(BinaryClassificationModel):
    def __init__(self,
                 num_units,
                 layer_num,
                 vocab_size,
                 embedding_size,
                 class_num,
                 learning_rate,
                 input_keep_prob=1.0,
                 output_keep_prob=1.0,
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
        self.input_keep_prob = input_keep_prob if is_train else 1.0
        self.output_keep_prob = output_keep_prob if is_train else 1.0
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
            cell_fw = rnn_factory(self.num_units, self.layer_num, self.cell_type, self.input_keep_prob,
                                  self.output_keep_prob)
            if self.bidirectional:
                cell_bw = rnn_factory(self.num_units, self.layer_num, self.cell_type, self.input_keep_prob,
                                      self.output_keep_prob)
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


    # def build(self):
    #     self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
    #     with tf.name_scope('input_placeholder'):
    #         self.input_sequences = tf.placeholder(tf.int32, [None, None], 'input_sequences')
    #         self.input_labels = tf.placeholder(tf.int32, [None], 'input_labels')
    #         self.input_lengths = tf.placeholder(tf.int32, [None], 'input_lengths')
    #
    #     with tf.device('/cpu:0'):
    #         self.embedding = init_embedding(self.vocab_size, self.embedding_size, self.embedding_path)
    #         input_sequences_emb = tf.nn.embedding_lookup(self.embedding, self.input_sequences)
    #
    #     with tf.variable_scope('Rnn'):
    #         cell_fw = rnn_factory(self.num_units, self.layer_num, self.cell_type, self.input_keep_prob,
    #                               self.output_keep_prob)
    #         if self.bidirectional:
    #             cell_bw = rnn_factory(self.num_units, self.layer_num, self.cell_type, self.input_keep_prob,
    #                                   self.output_keep_prob)
    #             rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
    #                                                              cell_bw,
    #                                                              input_sequences_emb,
    #                                                              sequence_length=self.input_lengths,
    #                                                              dtype=tf.float32)
    #             self.outputs = tf.concat(rnn_outputs, axis=2)
    #
    #         else:
    #             self.outputs, _ = tf.nn.dynamic_rnn(cell_fw, input_sequences_emb, dtype=tf.float32)
    #
    #         self.outputs_last_step = self.outputs[:, -1, :]
    #
    #     with tf.name_scope('output'):
    #         self.logist = tf.layers.dense(self.outputs_last_step, self.class_num)
    #         self.class_pro = tf.nn.softmax(self.logist, axis=1)
    #         self.class_label = tf.argmax(self.class_pro, axis=1, output_type=tf.int32)
    #         cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_labels, logits=self.logist)
    #         self.loss = tf.reduce_mean(cross_entropy)
    #         correct_pred = tf.equal(self.input_labels, self.class_label)
    #         self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #
    #     with tf.name_scope('Optimize'):
    #         if self.is_train:
    #             self.optimize = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
    #                                                                                            global_step=self.global_step)
    #
    #
    # def print_parms(self):
    #     print('%s : parms'%self.__class__.__name__)
    #     for var in tf.trainable_variables():
    #         print(var.name, var.shape)
