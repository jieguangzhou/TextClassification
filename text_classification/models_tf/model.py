import tensorflow as tf
from text_classification.nn_factory import init_embedding


class BinaryClassificationModel:
    def __init__(self, *args, **kwargs):
        self.build()

    def build(self):
        raise NotImplementedError

    def _build_placeholder(self):

        with tf.name_scope('input_placeholder'):
            self.input_sequences = tf.placeholder(tf.int32, [None, None], 'input_sequences')
            self.input_labels = tf.placeholder(tf.int32, [None], 'input_labels')
            self.input_lengths = tf.placeholder(tf.int32, [None], 'input_lengths')

    def _build_embedding(self, vocab_size, embedding_size, embedding_path):
        with tf.device('/cpu:0'):
            self.embedding = init_embedding(vocab_size, embedding_size, embedding_path)

    def _build_output(self, output, class_num):
        self.logist = tf.layers.dense(output, class_num)
        self.class_pro = tf.nn.softmax(self.logist, axis=1)
        self.class_label = tf.argmax(self.class_pro, axis=1, output_type=tf.int32)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_labels, logits=self.logist)
        self.loss = tf.reduce_mean(cross_entropy)
        correct_pred = tf.equal(self.input_labels, self.class_label)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def _build_optimize(self, loss, learning_rate, optimizer='adam'):
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        if optimizer.lower() == 'adam':
            Optimizer = tf.train.AdamOptimizer
        else:
            Optimizer = tf.train.GradientDescentOptimizer
        self.optimize = Optimizer(learning_rate=learning_rate).minimize(loss, global_step=self.global_step)

    def print_parms(self):
        print('%s : parms' % self.__class__.__name__)
        for var in tf.trainable_variables():
            print(var.name, var.shape)
