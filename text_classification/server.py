import argparse
import numpy as np
from flask import Flask, request
import json
from text_classification.main import use_backend
from text_classification.opt import add_train_opt, add_server_opt
from text_classification.utils import load_opt
from text_classification.data import load_vocab, UNK_IDX






def run(opt_model, opt_server, module):
    def deal_sentence(sentence):
        sequence = [vocab.get(char, UNK_IDX) for char in sentence]
        length = len(sequence)
        return np.array([sequence]), np.array([length])

    def predict(sentence):
        label_idxs, class_pros = model.inference(*deal_sentence(sentence))
        label_idx, class_pro = label_idxs[0], class_pros[0]

        label = idx2label[label_idx]
        pro = class_pro[label_idx]
        return label, pro

    _, model = module.create_model(opt_model, inference=True)
    model.to_inference(opt_model.save_path)

    vocab = load_vocab(opt_model.vocab_path)
    label2idx = load_vocab(opt_model.label_path)
    idx2label = {index:label for label, index in label2idx.items()}



    app = Flask('TextClassification')

    @app.route('/inference/')
    def inference():
        sentence = request.args.get("sentence")
        print(sentence)
        label, pro = predict(sentence)
        return json.dumps({'sentence':sentence, 'label':label, 'pro':float(pro)}, ensure_ascii=False)


    app.run('0.0.0.0', opt_server.port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_train_opt(parser)
    add_server_opt(parser)
    opt_server = parser.parse_args()
    save_path = opt_server.save_path
    opt_model = load_opt(save_path)
    module = use_backend(opt_model.backend)

    print(opt_model)
    print(opt_server)
    run(opt_model, opt_server, module)