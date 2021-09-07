import clip
import torch.nn as nn
from torchvision import datasets

def encode_siman_images(device='cuda:1'):
    model, prep = clip.load('ViT-B/32', jit=False, device)
    siman_ds = datasets.ImageFolder('simat_db/images')
    
    
def encode_siman_text():
    
    
def siman_eval(img_head, txt_head, emb_key='clip', lbds=[1], test=True, average=True, weight_power=.5,
              word_emb_type='single'):
    output = {}
    transfos = pd.read_csv('../siman/annotations/transfos4.csv', index_col=0)
    transfos = transfos[transfos.is_test == test]
    clip_siman = torch.load('/private/home/gcouairon/img2txt/eval/datasets/siman_processed/clip_siman.pt').float()
    img_embs = img_head(clip_siman)
    img_embs /= img_embs.norm(dim=-1, keepdim=True)
    value_embs = torch.stack([img_embs[did] for did in transfos.dataset_id])
    
    if word_emb_type == 'single':
        word_embs = dict(torch.load(f'data/siman_words_{emb_key}.ptd'))
        w2v = {k:txt_head(v.float()) for k, v in word_embs.items()}
        w2v = {k:v / v.norm(dim=-1, keepdim=True) for k, v in w2v.items()}
        delta_vectors = torch.stack([w2v[x.target] - w2v[x.value] for i, x in transfos.iterrows()])
    
    else:
        w2v = dict(torch.load(f'data/siman_dv2_{emb_key}.ptd'))
        delta_vectors = torch.stack([w2v[x.value, x.target] for i, x in transfos.iterrows()])
        
        
    #oscar_scores = torch.load('../eval/datasets/siman_processed/oscar_similarity_matrix.pt')
    oscar_scores = torch.load('../rebuttal/vilt_similarity_matrix.pt')
    weights = 1/np.array(transfos.norm2)**weight_power
    weights = weights/sum(weights)
    
    for lbd in lbds:
        target_embs = value_embs + lbd*delta_vectors

        nnb = (target_embs @ img_embs.T).topk(5).indices
        
        nnb_notself = [r[0] if r[0].item() != t else r[1] for r, t in zip(nnb, transfos.dataset_id)]
        
        scores = np.array([oscar_scores[ri, tc] for ri, tc in zip(nnb_notself, transfos.target_ids)]) > 3


        scores5 = torch.tensor([[oscar_scores[ri, tc] for ri, tc in zip(nnb[:, i], transfos.target_ids)] for i in range(5)]).max(0).values > 0.5
        
        if average:
            scores = 100*np.average(scores, weights=weights)
            scores5 = 100*np.average(scores5, weights=weights)
        
        #print(f'lbd={lbd:.1f}, scores={scores:.1f}, scores5={scores5:.2f}')
        output[lbd] = (scores, scores5)
    return output
    