from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
    """
    query and title for positive pairs.
    """
    def __init__(self, query, corpus, qrels, tokenizer, maxlen):
        self.corpus = corpus
        self.qrels = qrels
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.query = query

    def __len__(self):
        return len(self.query)

    def __getitem__(self, idx):
        q = self.query['title'].iloc[idx]
        pos_idx = self.qrels.iloc[idx].ravel()[0]
        pos = self.corpus['title'].loc[pos_idx]
        sample = self.tokenizer([q] + [pos], max_length=self.maxlen,truncation=True,padding='max_length',return_tensors='pt')
        return sample

class ValDataset(Dataset):
    """
    Used for validation, one sample includes a query, a positive title and 10 negative titles. 
    """
    def __init__(self, query, corpus, qrels, tokenizer, maxlen, neg_num, val_num):
        self.corpus = corpus
        self.qrels = qrels
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.query = query
        self.neg_num = neg_num
        self.val_num = val_num

    def __len__(self):
        return len(self.query)

    def __getitem__(self, idx):
        q = self.query['title'].iloc[idx]
        pos_idx = self.qrels.iloc[idx+(100000-self.val_num)].ravel()[0]
        pos = self.corpus['title'].loc[pos_idx]
        while True:
          neg_idx = np.random.randint(self.corpus.shape[0], size=self.neg_num)
          neg_idx = [x + 1 for x in neg_idx]
          if pos_idx not in neg_idx:
            break
        neg = self.corpus['title'].loc[neg_idx]
        sample = self.tokenizer([q] + [pos] + list(neg), max_length=self.maxlen,truncation=True,padding='max_length',return_tensors='pt')
        return sample

class TestDataset(Dataset):
    """
    Used to encode query and title embeddings.
    """
    def __init__(self, query, tokenizer, maxlen):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.query = query

    def __len__(self):
        return len(self.query)

    def __getitem__(self, idx):
        q = self.query['title'].iloc[idx]
        sample = self.tokenizer(q, max_length=self.maxlen,truncation=True,padding='max_length',return_tensors='pt')
        return sample

class NegDataset(Dataset):
    def __init__(self, query, corpus, qrels, neg_qrels, tokenizer, maxlen, phase='train', val_num=None):
        self.corpus = corpus
        self.qrels = qrels
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.query = query
        self.neg_qrels = neg_qrels
        assert phase in ['train', 'val']
        self.phase = phase
        self.val_num = val_num

    def __len__(self):
        return len(self.query)

    def __getitem__(self, idx):
        q = self.query['title'].iloc[idx]
        pos_idx = self.qrels.iloc[idx].ravel()[0] if self.phase=='train' else self.qrels.iloc[idx+(100000-self.val_num)].ravel()[0] 
        pos = self.corpus['title'].loc[pos_idx]
        neg_idx = self.neg_qrels[idx] if self.phase=='train' else self.neg_qrels[idx+(100000-self.val_num)]
        neg = self.corpus['title'].loc[neg_idx]
        sample = self.tokenizer([q] + [pos] + list(neg), max_length=self.maxlen,truncation=True,padding='max_length',return_tensors='pt')
        return sample

class NegValDataset(Dataset):
    """
    Used for validation, one sample includes a query, a positive title and 10 negative titles. 
    """
    def __init__(self, query, corpus, qrels, neg_qrels, tokenizer, maxlen, neg_num, val_num):
        self.corpus = corpus
        self.qrels = qrels
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.query = query
        self.neg_num = neg_num
        self.val_num = val_num
        self.neg_qrels = neg_qrels

    def __len__(self):
        return len(self.query)

    def __getitem__(self, idx):
        q = self.query['title'].iloc[idx]
        pos_idx = self.qrels.iloc[idx+(100000-self.val_num)].ravel()[0]
        pos = self.corpus['title'].loc[pos_idx]
        hard_neg_idx = self.neg_qrels[idx+(100000-self.val_num)]
        hard_neg = self.corpus['title'].loc[hard_neg_idx]
        while True:
          neg_idx = np.random.randint(self.corpus.shape[0], size=self.neg_num)
          neg_idx = [x + 1 for x in neg_idx]
          if pos_idx not in neg_idx:
            break
        neg = self.corpus['title'].loc[neg_idx]
        sample = self.tokenizer([q] + [pos] + list(hard_neg) + list(neg), max_length=self.maxlen,truncation=True,padding='max_length',return_tensors='pt')
        return sample
