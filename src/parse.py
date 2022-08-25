import argparse
import torch

def get_parse():

    parser = argparse.ArgumentParser()

    """# RetailRocket"""
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--dim', type=int, default=100, help='hidden state size')
    parser.add_argument('--epoch', type=int, default=20, help='the number of epochs to train for')
    parser.add_argument('--device', default='cuda', type=str,help='cuda or cpu')
    parser.add_argument('--topk', default=[10, 20], type=list)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate') 
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty') 
    
    args = parser.parse_args()
    if args.device == 'cuda':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: args.device = torch.device('cpu')
    return args
