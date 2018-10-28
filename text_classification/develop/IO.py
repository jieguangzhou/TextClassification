import os
import codecs
from logging import getLogger


WORK_PATH = os.getcwd()

def check_path(path, create=True):
    bool_exists = True
    logger = getLogger('develop.IO')
    if not os.path.exists(path):
        bool_exists = False
        if create:
            os.makedirs(path)
            logger.info('path is not exists, create %s' % path)
    return bool_exists


def read_file(file_path, mode='r', encoding='utf-8', deal_function=None, print_tqdm=False, tqdm_desc=None):
    r_f = codecs.open(file_path, mode=mode, encoding=encoding)
    if print_tqdm:
        from tqdm import tqdm
        line_num = int(os.popen('wc -l {}'.format(file_path)).readline().split()[0])
        r_f = tqdm(r_f, tqdm_desc, total=line_num)

    for line in r_f:
        if deal_function is not None:
            line = deal_function(line)
        yield line
    r_f.close()


def write_file(datas, file_path, mode='w', encoding='utf-8', deal_function=None, print_tqdm=False, tqdm_desc=None):
    if print_tqdm:
        from tqdm import tqdm
        datas = tqdm(datas, tqdm_desc)

    with codecs.open(file_path, mode=mode, encoding=encoding) as w_f:
        for line in datas:
            if deal_function is not None:
                line = deal_function(line)
            w_f.write(line)