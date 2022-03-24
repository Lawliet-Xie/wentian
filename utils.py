import scipy
import torch
import torch.nn.functional as F

def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


def compute_loss(y_pred, device, lamda=0.05):
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)  
    row = torch.arange(0,y_pred.shape[0],2, device=device)
    col = torch.arange(1,y_pred.shape[0],2, device=device)
    similarities = torch.index_select(similarities,0,row)
    similarities = torch.index_select(similarities,1,col)
    y_true = torch.arange(similarities.shape[0], device=device)
    similarities = similarities / lamda
    #论文中除以 temperature 超参 0.05
    loss = F.cross_entropy(similarities,y_true)
    return torch.mean(loss)

def compute_hard_loss(y_pred, device, lamda=0.05, batch_negative=True):
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)  
    row = torch.arange(0,y_pred.shape[0],6,device=device)
    col = torch.arange(y_pred.shape[0], device=device)
    similarities = torch.index_select(similarities,0,row)
    if batch_negative:
      col = torch.where(col % 6 != 0)[0].to(device)
      similarities = torch.index_select(similarities,1,col)
      y_true = torch.arange(0,len(col),5,device=device)
    else:
      col = torch.index_select(col.reshape(-1, 6).long().to(device), 1, torch.arange(1,6).to(device))
      similarities = similarities.gather(1, col)
      y_true = torch.zeros(len(row), device=device).long()
    print(similarities.shape)
    print(y_true)
    similarities = similarities / lamda
    #论文中除以 temperature 超参 0.05
    loss = F.cross_entropy(similarities,y_true)
    return torch.mean(loss)