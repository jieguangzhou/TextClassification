from collections import Counter
import logging
from random import shuffle
import numpy as np

from develop.IO import read_file, write_file

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

PAD_IDX = 0
UNK_IDX = 1


def load_vocab(vocab_path):
    vocab = {token: index for index, token in
             enumerate(read_file(vocab_path, deal_function=lambda x: x.strip() if x != '\n' else x))}
    logger.info('load vocab (size:%s) to  %s' % (len(vocab), vocab_path))
    return vocab


def save_vocab(vocab, vocab_path):
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    write_file(sorted_vocab, vocab_path, deal_function=lambda x: x[0] + '\n')
    logger.info('save vocab (size:%s) to  %s' % (len(vocab), vocab_path))


def load_data(data_path, vocab_path, label_vocab_path, create_vocab=False, create_label_vocab=False, min_freq=1,
              vocab_size=None):
    def deal_func(line):
        label, text = line.split('\t')
        tokens = list(filter(lambda x: x.strip(), text.split(' ')))
        return label, tokens

    vocab_ = Counter() if create_vocab else load_vocab(vocab_path)
    label_vocab = {} if create_label_vocab else load_vocab(label_vocab_path)

    labels, sequences, lengths = [], [], []
    msg = 'load data from  %s, ' % data_path
    for label, tokens in read_file(data_path, deal_function=deal_func):
        if create_vocab:
            vocab_.update(tokens)
        if create_label_vocab and label not in label_vocab:
            label_vocab[label] = len(label_vocab)

        labels.append(label)
        sequences.append(tokens)
        lengths.append(len(tokens))

    if create_vocab:
        vocab = {'<PAD>': PAD_IDX, '<UNK>': UNK_IDX}
        vocab_size = vocab_size or max(len(vocab_) - 2, 0)
        logger.info('create vocab, min freq: %s, vocab_size: %s' % (min_freq, vocab_size))
        for token, count in vocab_.most_common(vocab_size):
            if not token:
                continue
            if count < min_freq or len(vocab) >= vocab_size:
                break
            else:
                vocab[token] = len(vocab)
        save_vocab(vocab, vocab_path)
    else:
        vocab = vocab_

    if create_label_vocab:
        save_vocab(label_vocab, label_vocab_path)
    labels = [label_vocab[i] for i in labels]
    sequences = [[vocab.get(token, UNK_IDX) for token in sequence] for sequence in sequences]
    msg += 'total : %s' % len(labels)
    logger.info(msg)
    return np.array(sequences), np.array(labels), np.array(lengths)


def batch_iter(sequences, labels, lengths, batch_size=64, reverse=False, cut_length=None):
    data_num = len(labels)
    indexs = list(range(len(sequences)))
    shuffle(indexs)
    batch_start = 0
    shuffle_sequences = sequences[indexs]
    shuffle_labels = labels[indexs]
    shuffle_lengths = lengths[indexs]
    while batch_start < data_num:
        batch_end = batch_start + batch_size
        batch_sequences = shuffle_sequences[batch_start:batch_end]
        batch_labels = shuffle_labels[batch_start:batch_end]
        batch_lengths = shuffle_lengths[batch_start:batch_end]

        if isinstance(cut_length, int):
            batch_sequences = [sequence[:cut_length] for sequence in batch_sequences]
            batch_lengths = np.where(batch_lengths > cut_length, cut_length, batch_lengths)
        batch_max_length = batch_lengths.max()

        batch_padding_sequences = []
        for sequence, length in zip(batch_sequences, batch_lengths):
            sequence += [PAD_IDX] * (batch_max_length - length)
            if reverse:
                sequence.reverse()
            batch_padding_sequences.append(sequence)

        batch_padding_sequences = np.array(batch_padding_sequences)

        yield batch_padding_sequences, batch_labels, batch_lengths
        batch_start = batch_end


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    vocab_path = '../data/cnews/vocab.txt'
    label_vocab_path = '../data/cnews/label.txt'
    data_set = load_data('../data/cnews/cnews.val.txt.seg', vocab_path, label_vocab_path, vocab_size=5000)
    num = 0
    for sequences, labels, lengths in batch_iter(*data_set, batch_size=64):
        print(sequences.shape[1], lengths.max(), sequences.shape[1] == lengths.max())