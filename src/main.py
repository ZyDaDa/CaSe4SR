from tqdm import tqdm
from dataset import load_data
import torch
from parse import get_parse
from utils import fix_seed, topk_mrr_hr
from model import CaSe4SR

if __name__ == '__main__':
    fix_seed()
    args = get_parse()
    train_loader, test_loader, num_items, num_cat, cat4item = load_data(args)
    model = CaSe4SR(args, num_items, num_cat, cat4item)
    model.to(args.device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2)

    mini_batch = 0
    hr_max, mrr_max = {k:0 for k in args.topk}, {k:0 for k in args.topk} 
    for e in range(args.epoch):
        model.train()
        all_loss = 0.0
        bar = tqdm(train_loader, total=len(train_loader),ncols=100)
        for items, cats, item2idx, cat2idx, item_e, cat_e,item_e_w,cat_e_w, session_len, target in bar:
            
            scores = model(items.to(args.device), cats.to(args.device),
                            item2idx.to(args.device), cat2idx.to(args.device),
                            item_e.to(args.device), cat_e.to(args.device),
                            item_e_w.to(args.device), cat_e_w.to(args.device),
                            session_len  )

            optimizer.zero_grad() 
            loss = model.loss_function(scores, target.to(args.device)) 
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
            mini_batch += 1
            bar.set_postfix(Epoch=e, Train_Loss=loss.item()/target.size(0))
        print('epoch%d - loss%f'%(e,all_loss/len(train_loader)))
        
        model.eval()
        all_loss = 0.0
        hr, mrr = dict((k,0.0) for k in args.topk), dict((k,0.0) for k in args.topk) 
        test_num = 0
        for items, cats, item2idx, cat2idx, item_e, cat_e,item_e_w,cat_e_w, session_len, target in tqdm(test_loader,ncols=80,desc='test'):
            scores = model(items.to(args.device), cats.to(args.device),
                            item2idx.to(args.device), cat2idx.to(args.device),
                            item_e.to(args.device), cat_e.to(args.device),
                            item_e_w.to(args.device), cat_e_w.to(args.device),
                            session_len )
            loss = model.loss_function(scores, target.to(args.device)) 
            all_loss += loss.item()
            for k in hr.keys():
                this_hr, this_mrr = topk_mrr_hr(scores.detach().cpu(),target.numpy(),k)
                hr[k] += this_hr
                mrr[k] += this_mrr
            test_num += target.shape[0]

        for k in hr.keys():
            hr[k] /= test_num
            mrr[k] /= test_num
 
            if hr_max[k] < hr[k]: hr_max[k] = hr[k]
            if mrr_max[k] < mrr[k]: mrr_max[k] = mrr[k]
            print("best HR@%d %.2f\tMRR@%d %.2f"%(k,hr_max[k]*100,k,mrr_max[k]*100))
            print("now  HR@%d %.2f\tMRR@%d %.2f"%(k,hr[k]*100,k,mrr[k]*100))
            
    # Print the best score
    for k in args.topk:
        print('Top%d\thit%.2f\tmrr%.2f\tbest'%(k,hr_max[k]*100,mrr_max[k]*100))

        
