import torch
from torch.nn import Module, Parameter, Embedding, Linear, ReLU, \
    MultiheadAttention, LayerNorm, Dropout, Sequential
from torch.nn.init import kaiming_normal_


class SAKT(Module):
    """
        This implementation has a reference from: \
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer

        Args:
            number_questions: the total number of the questions(KCs) in the given dataset
            n: the length of the sequence of the questions or responses
            d: the dimension of the hidden vectors in this model
            number_attention_heads: the number of the attention heads in the \
                multi-head attention module in this model
            dropout: the dropout rate of this model
    """

    def __init__(self, number_questions, n, d, number_attention_heads, dropout):
        super().__init__()
        self.number_questions = number_questions
        self.n = n
        self.d = d
        self.number_attention_heads = number_attention_heads
        self.dropout = dropout

        self.M = Embedding(self.number_questions * 2, self.d)
        self.E = Embedding(self.number_questions, self.d)
        self.P = Parameter(torch.Tensor(self.n, self.d))

        kaiming_normal_(self.P)

        self.attn = MultiheadAttention(
            self.d, self.number_attention_heads, dropout=self.dropout
        )
        self.attn_dropout = Dropout(self.dropout)
        self.attn_layer_norm = LayerNorm(self.d)

        self.FFN = Sequential(
            Linear(self.d, self.d),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.d, self.d),
            Dropout(self.dropout),
        )
        self.FFN_layer_norm = LayerNorm(self.d)

        self.pred = Linear(self.d, 1)

    def forward(self, q, r, qry, rank):
        q = q.long()
        r = r.long()
        qry = qry.long()
        
        x = q + self.number_questions * r
        M = self.M(x).permute(1, 0, 2).to(rank)
        E = self.E(qry).permute(1, 0, 2).to(rank)
        P = self.P.unsqueeze(1).to(rank)

        causal_mask = torch.triu(
            torch.ones([E.shape[0], M.shape[0]]), diagonal=1
        ).bool().to(rank)

        M = M + P
        S, attn_weights = self.attn(E, M, M, attn_mask=causal_mask)
        S = self.attn_dropout(S)
        S = S.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        E = E.permute(1, 0, 2)
        S = self.attn_layer_norm(S + M + E)
        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)
        p = torch.sigmoid(self.pred(F)).squeeze()

        return p, attn_weights    
    