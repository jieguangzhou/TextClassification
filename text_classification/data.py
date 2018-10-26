from develop.IO import read_file, write_file
from collections import Counter

PAD_IDX = 0
UNK_IDX = 1


def load_vocab(vocab_path):
    vocab = {token: index for index, token in
             enumerate(read_file(vocab_path, deal_function=lambda x: x.strip() if x != '\n' else x))}
    return vocab


def save_vocab(vocab, vocab_path):
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    write_file(sorted_vocab, vocab_path, deal_function=lambda x: x[0] + '\n')


def load_data(data_path, vocab_path, label_vocab_path, create_vocab=False, create_label_vocab=False, min_freq=1,
              vocab_size=None):
    def deal_func(line):
        label, text = line.split('\t')
        tokens = list(filter(lambda x: x.strip(), text.split(' ')))
        return label, tokens

    vocab_ = Counter() if create_vocab else load_vocab(vocab_path)
    label_vocab = {} if create_label_vocab else load_vocab(label_vocab_path)

    labels, sequences, lengths = [], [], []
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
        print(vocab_size or max(len(vocab_)-2, 0))
        vocab_size = vocab_size or max(len(vocab_)-2, 0)
        for token, count in vocab_.most_common(vocab_size):
            if not token:
                continue
            if count < min_freq or len(vocab) >= vocab_size:
                break
            else:
                vocab[token] = len(vocab)
        save_vocab(vocab, vocab_path)
        print(len(vocab))
    else:
        vocab = vocab_

    if create_label_vocab:
        save_vocab(label_vocab, label_vocab_path)
        print(label_vocab)
    labels = [label_vocab[i] for i in labels]
    sequences = [[vocab.get(token, UNK_IDX) for token in sequence] for sequence in sequences]
    return sequences, labels, lengths


if __name__ == '__main__':
    vocab_path = '../data/cnews/vocab.txt'
    label_vocab_path = '../data/cnews/label.txt'
    sequences, labels, lengths = load_data('../data/cnews/cnews.val.txt.seg', vocab_path, label_vocab_path,
                                           create_vocab=True, create_label_vocab=True, vocab_size=1000)
    print(sequences[0])
    print(labels[0])
    print(lengths[0])
