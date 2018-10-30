import pickle
import os
def save_opt(opt, path):
    pickle.dump(opt, open(os.path.join(path, 'opt'), 'wb'))

def load_opt(path):
    opt = pickle.load(open(os.path.join(path, 'opt'), 'rb'))
    return opt
