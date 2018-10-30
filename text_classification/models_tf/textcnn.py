import tensorflow as tf
from text_classification.models_tf.model import BinaryClassificationModel


class TextCNN(BinaryClassificationModel):
    """
    TextCNN "Convolutional Neural Networks for Sentence Classification"
    """
    def __init__(self,
                 filter_num,
                 kernel_sizes,
                 sequence_length,
                 vocab_size,
                 embedding_size,
                 class_num,
                 learning_rate,
                 keep_drop_prob=1.0,
                 is_train=False,
                 embedding_path=None):

        self.filter_num = filter_num
        self.kernel_sizes = kernel_sizes
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.keep_drop_prob = keep_drop_prob if is_train else 1.0
        self.is_train = is_train
        self.embedding_path = embedding_path
        super(TextCNN, self).__init__()

    def build(self):
        self._build_placeholder()
        self._build_embedding(self.vocab_size, self.embedding_size, self.embedding_path)
        input_sequences_emb = tf.nn.embedding_lookup(self.embedding, self.input_sequences)
        input_sequences_emb = tf.expand_dims(input_sequences_emb, axis=-1)
        outputs = []
        with tf.variable_scope('CNN'):
            for kernel_size in self.kernel_sizes:
                conv = tf.layers.conv2d(input_sequences_emb, self.filter_num,
                                        kernel_size=[kernel_size, self.embedding_size])

                # 用reduce_max可以处理动态的序列长度，若用max_pooling则需要提供序列长度
                # maxpool = tf.layers.max_pooling2d(conv, pool_size=[self.sequence_length - kernel_size + 1, 1],
                #                                   strides=[1, 1])
                maxpool = tf.reduce_max(conv, axis=1)

                outputs.append(tf.squeeze(maxpool, axis=[1]))

        self.output = tf.concat(outputs, axis=1)

        self._build_output(self.output, self.class_num, self.keep_drop_prob)
        if self.is_train:
            self._build_optimize(self.loss, self.learning_rate)
