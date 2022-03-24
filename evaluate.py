from datasets import *
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertConfig
from model import Model
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from tqdm import tqdm

class Config:
    name = 'hard'
    seed = 2022
    root = '/public/home/yuqi/lawliet/nlp/wentian/'
    data_root = root + 'data/'
    model_name = root + "cache/chinese-roberta-wwm-ext/"
    pooling = 'first-last-avg'  #['cls','pooler','last-avg','first-last-avg']
    batch_size = 16
    val_num = 1000
    nepochs = 3
    lr = 3e-5
    maxlen = 64#80
    neg_num = 10
    hard_neg_num = 4 if name == 'hard' else None
    in_batch = True
    do_fgm = True
    current = "03-23-15-03"
    save_root = root + f'logs/{current}/'
    ckpt_root = save_root + 'ckpt/'
    output_root = save_root + 'output/'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dropout_rate = 0.1
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_config = BertConfig.from_pretrained(model_name)
    hidden_size = bert_config.hidden_size

cfg = Config
model = Model(cfg).to(cfg.device)
corpus_data = pd.read_csv(cfg.data_root + "corpus.tsv", sep="\t", names=["doc", "title"])
dev_data = pd.read_csv(cfg.data_root + "dev.query.txt", sep="\t", names=["query", "title"])
train_data = pd.read_csv(cfg.data_root + "train.query.txt", sep="\t", names=["query", "title"])
qrels = pd.read_csv(cfg.data_root + "qrels.train.tsv", sep="\t", names=["query", "doc"])
corpus_data = corpus_data.set_index("doc")
dev_data = dev_data.set_index("query")
train_data = train_data.set_index("query")
qrels = qrels.set_index("query")


state_dict = torch.load(cfg.ckpt_root + 'best_simcse.pth', map_location=str(cfg.device))
model.load_state_dict(state_dict)
dev_dataset = TestDataset(dev_data, cfg.bert_tokenizer, cfg.maxlen)
corpus_dataset = TestDataset(corpus_data, cfg.bert_tokenizer, cfg.maxlen)
dev_dataloader = DataLoader(dev_dataset, batch_size=16, num_workers=4, shuffle=False)
corpus_dataloader = DataLoader(corpus_dataset, batch_size=16, num_workers=4, shuffle=False)

dev_embeddings = []
doc_embeddings = []
model.eval()

for data in tqdm(dev_dataloader):
    input_ids = data['input_ids'][:,0,:].to(cfg.device)
    attention_mask = data['attention_mask'][:,0,:].to(cfg.device)
    token_type_ids = data['token_type_ids'][:,0,:].to(cfg.device)
    query_embed = model(input_ids, attention_mask, token_type_ids)
    dev_embeddings.append(query_embed.detach().cpu())

dev_embeddings = torch.cat(dev_embeddings, dim=0).numpy()
dev_embeddings = normalize(dev_embeddings)
with open(cfg.output_root+'query_embedding', 'w') as up:
    for id, feat in zip(dev_data.index, dev_embeddings):
        up.write('{0}\t{1}\n'.format(id, ','.join([str(round(x,8)) for x in feat])))

for data in tqdm(corpus_dataloader):
    input_ids = data['input_ids'][:,0,:].to(cfg.device)
    attention_mask = data['attention_mask'][:,0,:].to(cfg.device)
    token_type_ids = data['token_type_ids'][:,0,:].to(cfg.device)
    doc_embed = model(input_ids, attention_mask, token_type_ids)
    doc_embeddings.append(doc_embed.detach().cpu())

doc_embeddings = torch.cat(doc_embeddings, dim=0).numpy()
doc_embeddings = normalize(doc_embeddings)
with open(cfg.output_root+'doc_embedding', 'w') as up:
    for id, feat in zip(corpus_data.index, doc_embeddings):
        up.write('{0}\t{1}\n'.format(id, ','.join([str(round(x,8)) for x in feat])))