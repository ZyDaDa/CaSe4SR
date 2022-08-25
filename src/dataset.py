from lib2to3.pgen2.tokenize import TokenError
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import pickle
from tqdm import tqdm
import networkx as nx
import numpy as np

def load_data(args):
    dataset_folder = os.path.abspath(os.path.join('dataset'))

    train_set = SeqDataset(dataset_folder,'train')
    test_set = SeqDataset(dataset_folder,'test')

    train_loader = DataLoader(train_set,args.batch_size,  num_workers=0,
                              shuffle=True,collate_fn=collate_fn)
    test_loader = DataLoader(test_set,args.batch_size, num_workers=0,
                              shuffle=False,collate_fn=collate_fn)

    id_maps = pickle.load(open(os.path.join( dataset_folder,'idmap.pkl'), 'rb'))
    item_num = max(id_maps[0].values())+1
    cat_num = max(id_maps[1].values())+1

    # get category of item
    cat4item = [0]*item_num
    for d in pickle.load(open(os.path.join(dataset_folder, 'train.pkl'),'rb')):
        for i,c in zip(d['items'],d['cats']):
            cat4item[i] = c
    
    return train_loader, test_loader, item_num, cat_num, cat4item

class SeqDataset(Dataset):
    def __init__(self, datafolder, file='train',max_len=50) -> None:
        super().__init__()
        self.max_len = max_len
        data_file = os.path.join(datafolder, file+'.pkl')

        self.data = pickle.load(open(data_file,'rb')) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index): 
        # get raw data
        session = self.data[index]['items'][-self.max_len:]
        cats = self.data[index]['cats'][-self.max_len:]
        target = self.data[index]['target']
        # construct graph
        item_node = np.unique(session)
        item_map = dict([(i,idx) for idx,i in enumerate(item_node)]) # pyg data format

        cat_node = np.unique(cats)
        cat_map = dict([(i,idx) for idx,i in enumerate(cat_node)]) # pyg data format

        item2idx = [item_map[i] for i in session] # convert itemid to index in item_node
        cat2idx = [cat_map[i] for i in cats] # same to above

        item_out_deg = {} # out degree of item node
        cat_out_dge = {} # out degree of category node
        for i in item2idx[:-1]:
            item_out_deg[i] = item_out_deg.get(i,0) + 1
        for i in cat2idx[:-1]:
            cat_out_dge[i] = cat_out_dge.get(i,0) + 1

        item_in_deg = {} # out degree of item node
        cat_in_dge = {} # out degree of category node
        for i in item2idx[1:]:
            item_in_deg[i] = item_in_deg.get(i,0) + 1
        for i in cat2idx[1:]:
            cat_in_dge[i] = cat_in_dge.get(i,0) + 1

        item_edge = [[],[]] 
        cat_edge = [[],[]]

        # compute edge weight
        item_in_edge_weight = []
        item_out_edge_weight = []
        cat_in_edge_weight = []
        cat_out_edge_weight = []
        for h,t in zip(item2idx[:-1],item2idx[1:]):
            item_edge[0].append(h)
            item_edge[1].append(t)
            item_in_edge_weight.append(1/item_in_deg[t])
            item_out_edge_weight.append(1/item_out_deg[h])
        for h,t in zip(cat2idx[:-1],cat2idx[1:]):
            cat_edge[0].append(h)
            cat_edge[1].append(t)
            cat_in_edge_weight.append(1/cat_in_dge[t])
            cat_out_edge_weight.append(1/cat_out_dge[h])
        item_dege_weight = [item_in_edge_weight,item_out_edge_weight]
        cat_edge_weight = [cat_in_edge_weight, cat_out_edge_weight]

        return item_node, cat_node, item2idx, cat2idx, item_edge, cat_edge, item_dege_weight, cat_edge_weight, target

def collate_fn(batch_data):
    batch_item_nodes = [] # 1d 
    batch_cat_nodes = [] # 1d
    batch_item2idx = [] # 1d
    batch_cat2idx = [] # 1d
    batch_item_edge = [] # n*2
    batch_cat_edge = [] # n*2

    batch_item_edge_weight = []
    batch_cat_edge_weight = []

    batch_session_len = [] # 1d , split above tensor by this term

    batch_target = []

    now_item_idx = 0
    now_cat_idx = 0
    for d in batch_data:
        item_node, cat_node, item2idx, cat2idx, item_edge, cat_edge, item_dege_weight, cat_edge_weight, target = d

        batch_item_nodes.append(torch.LongTensor(item_node))
        batch_cat_nodes.append(torch.LongTensor(cat_node))

        batch_item2idx.append(torch.LongTensor(item2idx)+now_item_idx)
        batch_cat2idx.append(torch.LongTensor(cat2idx)+now_cat_idx)

        batch_item_edge.append(torch.LongTensor(item_edge)+now_item_idx)
        batch_cat_edge.append(torch.LongTensor(cat_edge)+now_cat_idx)

        batch_item_edge_weight.append(torch.FloatTensor(item_dege_weight))
        batch_cat_edge_weight.append(torch.FloatTensor(cat_edge_weight))

        batch_session_len.append(len(item2idx))
        batch_target.append(target)

        now_item_idx += len(item_node)
        now_cat_idx += len(cat_node)

    batch_item_nodes = torch.concat(batch_item_nodes)
    batch_cat_nodes = torch.concat(batch_cat_nodes)
    batch_item2idx = torch.concat(batch_item2idx)
    batch_cat2idx = torch.concat(batch_cat2idx)

    batch_item_edge = torch.concat(batch_item_edge,dim=1)
    batch_cat_edge = torch.concat(batch_cat_edge,dim=1)

    batch_target = torch.LongTensor(batch_target)

    batch_item_edge_weight = torch.concat(batch_item_edge_weight, dim=-1)
    batch_cat_edge_weight = torch.concat(batch_cat_edge_weight,dim=-1)

    return batch_item_nodes, batch_cat_nodes, batch_item2idx, batch_cat2idx, batch_item_edge, batch_cat_edge, batch_item_edge_weight,batch_cat_edge_weight, batch_session_len, batch_target