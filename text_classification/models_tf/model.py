import tensorflow as tf
from text_classification.models_tf.nn_factory import init_embedding


class BinaryClassificationModel:
    def __init__(self, *args, **kwargs):
        self.build()
        self.name = self.__class__.__name__

    def build(self):
        raise NotImplementedError

    def inference(self, sequences, lengths):
        labels, class_pro = self.session.run([self.class_label, self.class_pro],
                                             feed_dict={self.input_sequences: sequences,
                                                        self.input_lengths: lengths})

        return labels, class_pro

    def to_inference(self, model_path):
        ckpt = tf.train.get_checkpoint_state(model_path)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(self.session, ckpt.model_checkpoint_path)
        print('use_inference')

    def _build_placeholder(self, batch_size=None, sequence_length=None):
        """
        构建placeholder
        :param batch_size: 默认为None,即动态batch_size
        :param sequence_length: 序列长度
        """
        with tf.name_scope('input_placeholder'):
            self.input_sequences = tf.placeholder(tf.int32, [batch_size, sequence_length], 'input_sequences')
            self.input_labels = tf.placeholder(tf.int32, [batch_size], 'input_labels')
            self.input_lengths = tf.placeholder(tf.int32, [batch_size], 'input_lengths')

    def _build_embedding(self, vocab_size, embedding_size, embedding_path):
        with tf.device('/cpu:0'):
            self.embedding = init_embedding(vocab_size, embedding_size, embedding_path)

    def _build_output(self, output, class_num, keep_drop_prob=1.0):
        with tf.name_scope('dropout'):
            output = tf.nn.dropout(output, keep_drop_prob)

        # 将output维度转为类别数量
        self.logist = tf.layers.dense(output, class_num)
        self.class_pro = tf.nn.softmax(self.logist, axis=1)
        self.class_label = tf.argmax(self.class_pro, axis=1, output_type=tf.int32)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_labels, logits=self.logist)
        self.loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', self.loss)

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
        print('\n', '-' * 20)
        print('%s : parms' % self.name)
        for var in tf.trainable_variables():
            print(var.name, var.shape)
        print('-' * 20, '\n')
