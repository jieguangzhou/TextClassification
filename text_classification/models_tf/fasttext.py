import tensorflow as tf
from text_classification.models_tf.model import BinaryClassificationModel

class FastText(BinaryClassificationModel):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 class_num,
                 learning_rate,
                 keep_drop_prob=1.0,
                 is_train=False,
                 embedding_path=None):

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.keep_drop_prob = keep_drop_prob if is_train else 1.0
        self.is_train = is_train
        self.embedding_path = embedding_path
        super(FastText, self).__init__()

    def build(self):
        self._build_placeholder()
        self._build_embedding(self.vocab_size, self.embedding_size, self.embedding_path)
        input_sequences_emb = tf.nn.embedding_lookup(self.embedding, self.input_sequences)

        self.output = tf.reduce_mean(input_sequences_emb, axis=1)

        self._build_output(self.output, self.class_num, self.keep_drop_prob)
        if self.is_train:
            self._build_optimize(self.loss, self.learning_rate)