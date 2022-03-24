import time
import torch
import logging
import os
import numpy as np
import random
import yaml
from transformers import BertTokenizer, BertConfig

def setup(Config):
    if Config.seed is not None:
        set_seed(Config.seed)
    
    for path in [Config.root, Config.data_root, Config.save_root, Config.ckpt_root, Config.output_root]:
        os.makedirs(path, exist_ok=True)
    
    create_logger(Config.save_root)
    # other change of config

    log_cfg(Config)
    return Config


class Config:
    name = 'hard'
    seed = 2022
    root = '/public/home/yuqi/lawliet/nlp/wentian/'
    data_root = root + 'data/'
    model_name = root + "cache/chinese-roberta-wwm-ext/"
    pooling = 'first-last-avg'  #['cls','pooler','last-avg','first-last-avg']
    batch_size = 32
    val_num = 1000
    nepochs = 3
    lr = 3e-5
    maxlen = 64#80
    neg_num = 10
    hard_neg_num = 4 if name == 'hard' else None
    in_batch = True
    do_fgm = True
    current = time.strftime('%m-%d-%H-%M')
    save_root = root + f'logs/{current}/'
    ckpt_root = save_root + 'ckpt/'
    output_root = save_root + 'output/'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dropout_rate = 0.1
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_config = BertConfig.from_pretrained(model_name)
    hidden_size = bert_config.hidden_size



def create_logger(log_dir):
    # create logger
    os.makedirs(log_dir, exist_ok=True)
    log_file = 'logging.log'
    final_log_file = os.path.join(log_dir, log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = '[%(asctime)s] %(message)s'

    file = logging.FileHandler(filename=final_log_file, mode='a')
    file.setLevel(logging.INFO)
    file.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file)

    return logger

def set_seed(seed):
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_cfg(Config):
    cfg_dict = {}
    for name in dir(Config):
        if not name.startswith('__') and 'bert' not in name and name not in ['device']:
            cfg_dict[name] = getattr(Config, name)
    logging.info('===cfg===\n' +  yaml.safe_dump(cfg_dict, indent=4))


if __name__ == '__main__':
    cfg = setup(Config)


