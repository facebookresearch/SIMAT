# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

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
    heads = torch.load(f'data/head_{emb_key}_t={args.tau}.pt')
    #heads = dict(img_head = lambda x:x, txt_head=lambda x:x)
    output = {}
    transfos = pd.read_csv('simat_db/transfos.csv', index_col=0)
    triplets = pd.read_csv('simat_db/triplets.csv', index_col=0)
    did2rid = dict(zip(triplets.dataset_id, triplets.index))
    rid2did = dict(zip(triplets.index, triplets.dataset_id))
    
    transfos = transfos[transfos.is_test == (args.domain == 'test')]
    
    transfos_did = [rid2did[rid] for rid in transfos.region_id]
    
    #new method
    clip_simat = torch.load('data/simat_img_clip.pt')
    img_embs_stacked = torch.stack([clip_simat[did2rid[i]] for i in range(len(clip_simat))]).float()
    img_embs_stacked = heads['img_head'](img_embs_stacked).normalize()
    value_embs = torch.stack([img_embs_stacked[did] for did in transfos_did])
    
    
    word_embs = dict(torch.load(f'data/simat_words_{emb_key}.ptd'))
    w2v = {k:heads['txt_head'](v.float()).normalize() for k, v in word_embs.items()}
    delta_vectors = torch.stack([w2v[x.target] - w2v[x.value] for i, x in transfos.iterrows()])
    
    oscar_scores = torch.load('simat_db/oscar_similarity_matrix.pt')
    weights = 1/np.array(transfos.norm2)**.5
    weights = weights/sum(weights)
    
    for lbd in args.lbds:
        target_embs = value_embs + lbd*delta_vectors

        nnb = (target_embs @ img_embs_stacked.T).topk(5).indices
        nnb_notself = [r[0] if r[0].item() != t else r[1] for r, t in zip(nnb, transfos_did)]
        
        scores = np.array([oscar_scores[ri, tc] for ri, tc in zip(nnb_notself, transfos.target_ids)]) > .5

        
        output[lbd] = 100*np.average(scores, weights=weights)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run eval')
    parser.add_argument('--domain', type=str, default='dev', help='domain, test or dev')
    parser.add_argument('--backbone', type=str, default='clip', help='backbone method. Only clip is supported.')
    parser.add_argument('--tau', type=float, default=0.1, help='pretraining temperature tau')
    parser.add_argument('--lbds', nargs='+', default=[1], help='list of values for lambda')
    args = parser.parse_args()
    args.lbds = [float(l) for l in args.lbds]
    
    output = simat_eval(args)
    print('SIMAT Scores:')
    for lbd, v in output.items():
        print(f'{lbd=}: {v:.2f}')