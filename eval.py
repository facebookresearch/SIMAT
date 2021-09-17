import clip
import torch.nn as nn
from torchvision import datasets
import argparse
import torch
import pandas as pd
import numpy as np

torch.Tensor.normalize = lambda x: x/x.norm(dim=-1, keepdim=True)
    
def simat_eval(args):
    #img_head, txt_head, emb_key='clip', lbds=[1], test=True:, tau
    # get heads !
    emb_key = 'clip'
    heads = torch.load(f'data/head_{emb_key}_t={args.t}.pt')
    output = {}
    transfos = pd.read_csv('simat_db/transfos.csv', index_col=0)
    transfos = transfos[transfos.is_test == (args.domain == 'test')]
    clip_simat = torch.load('data/clip_simat.pt').float()
    img_embs = heads['img_head'](clip_simat).normalize()
    value_embs = torch.stack([img_embs[did] for did in transfos.dataset_id])
    
    word_embs = dict(torch.load(f'data/simat_words_{emb_key}.ptd'))
    w2v = {k:heads['txt_head'](v.float()).normalize() for k, v in word_embs.items()}
    delta_vectors = torch.stack([w2v[x.target] - w2v[x.value] for i, x in transfos.iterrows()])
    
    oscar_scores = torch.load('simat_db/oscar_similarity_matrix.pt')
    weights = 1/np.array(transfos.norm2)**.5
    weights = weights/sum(weights)
    
    for lbd in args.lbds:
        target_embs = value_embs + lbd*delta_vectors

        nnb = (target_embs @ img_embs.T).topk(5).indices
        
        nnb_notself = [r[0] if r[0].item() != t else r[1] for r, t in zip(nnb, transfos.dataset_id)]
        
        scores = np.array([oscar_scores[ri, tc] for ri, tc in zip(nnb_notself, transfos.target_ids)]) > .5

        
        output[lbd] = 100*np.average(scores, weights=weights)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run eval')
    parser.add_argument('--domain', type=str, default='dev', help='domain, test or dev')
    parser.add_argument('--backbone', type=str, default='clip', help='backbone method. Only clip is supported.')
    parser.add_argument('--t', type=float, default=0.1, help='pretraining temperature tau')
    parser.add_argument('--lbds', nargs='+', default=[1], help='list of values for lambda')
    args = parser.parse_args()
    args.lbds = [float(l) for l in args.lbds]
    
    output = simat_eval(args)
    print('SIMAT Scores:')
    for lbd, v in output.items():
        print(f'{lbd=}: {v:.2f}')
    