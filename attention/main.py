import torch
import math
# softmax output numbers that add up to one. num/total
# softmax(QK^T / sqrt(d_k))V
def attention(Q, K, V): # Query Key Value
    d_k = Q.shape[-1]
    # lets say input is "i like cats", (3, 8) 3 tokens, embbeding vector of size 8, (usually it's 512 or 788)
    # Q = (3, 8) — "what each token is looking for"
    # K = (3, 8) — "what each token has to offer, as a label"
    # V = (3, 8) — "what each token actually contains, if picked"
    cur = (torch.matmul(Q, K.transpose(-2, -1))) / torch.sqrt(torch.tensor(d_k)) * V
    return torch.softmax(cur, dim=0)