import torch
import torch.nn as nn
import math
from torch import Tensor

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)

class PositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(PositionEmbedding, self).__init__()
        #TODO
        # position = torch.arange(max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        # _2i = torch.arange(0, d_model, 2, dtype=torch.float)
        # div_term = 10000 ** (_2i / d_model)
        # self.encoding = torch.zeros(max_len, d_model)
        # self.encoding[:, 0::2] = torch.sin(position / div_term)
        # self.encoding[:, 1::2] = torch.cos(position / div_term)
        # self.encoding = self.encoding.unsqueeze(0)

        pos_embedding = torch.zeros((max_len, d_model))
        den = torch.exp(-torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.register_buffer('pos_embedding', pos_embedding)   ## (kyu) update 되지 않도록
    
    def forward(self, x: Tensor) -> Tensor:
        #TODO one line!
        # batch_size, seq_len = x.size()
        # return self.encoding[:, :seq_len].to(x.device)

        return self.pos_embedding[:x.size(-2), :]