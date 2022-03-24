from configs import setup, Config
from datasets import *
import pandas as pd
from model import Model, FGM
import torch
from torch.utils.data import DataLoader
import logging
from utils import *
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
import os
import argparse

def validation(val_dataloader, model, cfg):
    corref_list = []
    with torch.no_grad():
        for data in val_dataloader:
            query_input_ids = data['input_ids'][:,0,:].to(cfg.device)
            query_attention_mask = data['attention_mask'][:,0,:].to(cfg.device)
            query_token_type_ids = data['token_type_ids'][:,0,:].to(cfg.device)
            query_pred = model(query_input_ids,query_attention_mask,query_token_type_ids)
            query_pred = query_pred.unsqueeze(1)
            doc_input_ids = data['input_ids'][:,1:,:].reshape(len(data['input_ids'])*(cfg.hard_neg_num+1),-1).to(cfg.device)
            doc_attention_mask = data['attention_mask'][:,1:,:].reshape(len(data['attention_mask'])*(cfg.hard_neg_num+1),-1).to(cfg.device)
            doc_token_type_ids = data['token_type_ids'][:,1:,:].reshape(len(data['token_type_ids'])*(cfg.hard_neg_num+1),-1).to(cfg.device)
            doc_pred = model(doc_input_ids,doc_attention_mask,doc_token_type_ids)
            doc_pred = doc_pred.reshape(len(data['input_ids']), cfg.hard_neg_num+1, -1)
            similarity = F.cosine_similarity(query_pred,doc_pred, dim=-1)
            # label = np.zeros(similarity.shape)
            # label[:,0] = 1
            # similarity = similarity.cpu().detach().numpy()
            # for i in range(label.shape[0]):
            #     corrcoef = compute_corrcoef(label[i], similarity[i])
            #     corref_list.append(corrcoef)
            preds = torch.argmax(similarity,dim=-1).long().cpu().numpy()
            labels = torch.zeros(similarity.size(0)).long().cpu().numpy()
            result = accuracy_score(labels,preds)
            corref_list.append(result)
    return np.mean(corref_list)  

           
def train(train_dataloader, val_dataloader, model, optimizer, cfg):
    model.train()
    size = len(train_dataloader)
    max_corref = 0
    if cfg.do_fgm:
        fgm = FGM(model)
        name = 'word_embeddings.weight'
    for epoch in range(cfg.nepochs):
        for batch, data in enumerate(train_dataloader):
            input_ids = data['input_ids'].view(len(data['input_ids'])*(cfg.hard_neg_num+2),-1).to(cfg.device)
            attention_mask = data['attention_mask'].view(len(data['attention_mask'])*(cfg.hard_neg_num+2),-1).to(cfg.device)
            token_type_ids = data['token_type_ids'].view(len(data['token_type_ids'])*(cfg.hard_neg_num+2),-1).to(cfg.device)
            pred = model(input_ids,attention_mask,token_type_ids)
            optimizer.zero_grad()
            loss = compute_hard_loss(pred, cfg.device, batch_negative=cfg.in_batch)
            loss.backward()
            # do fgm
            if cfg.do_fgm:
                fgm.attack(emb_name=name)
                pred = model(input_ids,attention_mask,token_type_ids)
                loss_adv = compute_hard_loss(pred, cfg.device, batch_negative=cfg.in_batch)
                loss_adv.backward()
                fgm.restore(name)
            
            optimizer.step()
            if batch % 150 == 0:
                loss= loss.detach().item()
                model.eval()
                corref = validation(val_dataloader, model, cfg)
                logging.info(f"[Epoch {epoch} | {batch:>5d}/{size:>5d}] | loss: {loss:>7f} | corref: {corref:>4f}")
                if corref > max_corref:
                    max_corref = corref
                    torch.save(model.state_dict(),cfg.ckpt_root+'best_simcse.pth')
                    logging.info(f"Higher corrcoef: {corref:>4f}, model saved to best_simcse.pth")
                model.train()
        torch.save(model.state_dict(),cfg.ckpt_root + 'latest_simcse.pth')
        logging.info("Model saved to latest_simcse.pth")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None)
    args = parser.parse_args()
    return args

def main(cfg):
    #args = parse_args()
    corpus_data = pd.read_csv(cfg.data_root + "corpus.tsv", sep="\t", names=["doc", "title"])
    dev_data = pd.read_csv(cfg.data_root + "dev.query.txt", sep="\t", names=["query", "title"])
    train_data = pd.read_csv(cfg.data_root + "train.query.txt", sep="\t", names=["query", "title"])
    qrels = pd.read_csv(cfg.data_root + "qrels.train.tsv", sep="\t", names=["query", "doc"])
    corpus_data = corpus_data.set_index("doc")
    dev_data = dev_data.set_index("query")
    train_data = train_data.set_index("query")
    qrels = qrels.set_index("query")
    neg_qrels = np.load(cfg.data_root + 'neg_qrels.npy')
    assert cfg.hard_neg_num == neg_qrels.shape[1]

    train_dataset = NegDataset(train_data[:-cfg.val_num], corpus_data, qrels, neg_qrels, cfg.bert_tokenizer, cfg.maxlen, phase='train')
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=4, shuffle=True)
    val_dataset = NegValDataset(train_data[-cfg.val_num:], corpus_data, qrels, neg_qrels, cfg.bert_tokenizer, cfg.maxlen, neg_num=cfg.neg_num, val_num=cfg.val_num)
    val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=4)
    logging.info(f'==> Length of train dataset: {len(train_dataset)}')
    logging.info(f'==> Length of validation dataset: {len(val_dataset)}')

    model = Model(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    # load the model from first stage
    #resume_path = cfg.root + f'logs/{args.path}/ckpt/'
    #logging.info(f'==> loading the first stage model from {resume_path}')
    #state_dict = torch.load(resume_path + 'best_simcse.pth', map_location=str(cfg.device))
    #model.load_state_dict(state_dict)
    train(train_dataloader, val_dataloader, model, optimizer, cfg)

    # evalutation
    state_dict = torch.load(cfg.ckpt_root + 'best_simcse.pth', map_location=str(cfg.device))
    model.load_state_dict(state_dict)
    dev_dataset = TestDataset(dev_data, cfg.bert_tokenizer, cfg.maxlen)
    corpus_dataset = TestDataset(corpus_data, cfg.bert_tokenizer, cfg.maxlen)
    dev_dataloader = DataLoader(dev_dataset, batch_size=16, num_workers=4, shuffle=False)
    corpus_dataloader = DataLoader(corpus_dataset, batch_size=16, num_workers=4, shuffle=False)

    dev_embeddings = []
    doc_embeddings = []
    model.eval()

    for data in dev_dataloader:
        input_ids = data['input_ids'][:,0,:].to(cfg.device)
        attention_mask = data['attention_mask'][:,0,:].to(cfg.device)
        token_type_ids = data['token_type_ids'][:,0,:].to(cfg.device)
        query_embed = model(input_ids, attention_mask, token_type_ids)
        dev_embeddings.append(query_embed.detach().cpu())

    for data in corpus_dataloader:
        input_ids = data['input_ids'][:,0,:].to(cfg.device)
        attention_mask = data['attention_mask'][:,0,:].to(cfg.device)
        token_type_ids = data['token_type_ids'][:,0,:].to(cfg.device)
        doc_embed = model(input_ids, attention_mask, token_type_ids)
        doc_embeddings.append(doc_embed.detach().cpu())

    dev_embeddings = torch.cat(dev_embeddings, dim=0).numpy()
    doc_embeddings = torch.cat(doc_embeddings, dim=0).numpy()
    dev_embeddings = normalize(dev_embeddings)
    doc_embeddings = normalize(doc_embeddings)
    logging.info(f'==> The shape of dev_embeddings: {dev_embeddings.shape}')
    logging.info(f'==> The shape of dev_embeddings: {doc_embeddings.shape}')

    with open(cfg.output_root+'query_embedding', 'w') as up:
        for id, feat in zip(dev_data.index, dev_embeddings):
            up.write('{0}\t{1}\n'.format(id, ','.join([str(round(x,8)) for x in feat])))
        
    with open(cfg.output_root+'doc_embedding', 'w') as up:
        for id, feat in zip(corpus_data.index, doc_embeddings):
            up.write('{0}\t{1}\n'.format(id, ','.join([str(round(x,8)) for x in feat])))

    logging.info(f'==> Finished.')



if __name__ == '__main__':
    cfg = setup(Config)
    os.chdir(cfg.root)
    main(cfg)
