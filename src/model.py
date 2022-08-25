import torch
from torch import nn
import numpy as np
import math
from GNN import GNN

class CaSe4SR(nn.Module):
    def __init__(self,args, num_item, num_cat, cat4item) -> None:
        super().__init__()
        self.dim = args.dim

        self.item_embedding = nn.Embedding(num_item, args.dim)
        self.cat_embedding = nn.Embedding(num_cat, args.dim)

        self.cat4item = nn.parameter.Parameter(torch.LongTensor(cat4item),requires_grad=False)

        self.item_gnn = GNN(args.dim,args.dim)
        self.cat_gnn = GNN(args.dim,args.dim)

        self.q = nn.Linear(args.dim*2,1)
        self.w1 = nn.Linear(args.dim*2, args.dim*2)
        self.w2 = nn.Linear(args.dim*2, args.dim*2,bias=False)

        self.loss_function = nn.CrossEntropyLoss()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=stdv)
        nn.init.normal_(self.item_embedding.weight,std=stdv)
        nn.init.normal_(self.cat_embedding.weight,std=stdv)

    def forward(self, items, cats, item2idx, cat2idx, item_e, cat_e,item_e_w,cat_e_w, session_len ):

        item_emb = self.item_embedding(items)
        cat_emb = self.cat_embedding(cats)

        et = self.item_gnn(item_emb, item_e, item_e_w)[item2idx]
        ft = self.cat_gnn(cat_emb, cat_e, cat_e_w)[cat2idx]

        eft = torch.concat([et,ft],-1)
        per_sess_emb = torch.split(eft,session_len) # split tensor by session length

        last_emb = torch.concat([embs[-1].unsqueeze(0).repeat(len(embs),1) for embs in per_sess_emb],dim=0) # [en;fn]

        alpha = self.q(torch.sigmoid(self.w1(eft) + self.w2(last_emb)))

        per_final_emb = torch.split((alpha*eft),session_len) # split tensor by session length

        sg = torch.stack([embs.sum(0) for embs in per_final_emb],dim=0)

        all_item = torch.concat([self.item_embedding.weight, self.cat_embedding(self.cat4item)],dim=-1)
        return torch.matmul(sg, all_item.T)
