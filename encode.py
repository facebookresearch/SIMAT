import clip
import torch
import torchvision.datasets as datasets
from functools import partial
from tqdm import tqdm
import pandas as pd

# code for encoding the SIMAT database with CLIP
# produces the files clip_simat.pt and simat_words_clip.ptd

device = 'cuda:0'
device = 'cpu'

model, prep = clip.load('ViT-B/32', device=device, jit=False)

ds = datasets.ImageFolder('simat_db/images', transform=prep)

dl = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=10, shuffle=False)

img_enc = torch.cat([model.encode_image(b.to(device)).cpu().detach() for b, i in tqdm(dl)])
img_enc /= img_enc.norm(dim=-1, keepdim=True)
torch.save(img_enc, 'data/clip_simat_2.pt')

# encode words
transfos = pd.read_csv('simat_db/transfos.csv', index_col=0)
words = list(set(transfos.target) | set(transfos.value))
tokens = clip.tokenize(words)

word_encs = torch.cat([model.encode_text(b.to(device)).cpu().detach() for b in tqdm(tokens.split(32))])
word_encs /= word_encs.norm(dim=-1, keepdim=True)

w2we = dict(zip(words, word_encs))
torch.save(w2we, 'data/simat_words_clip_2.ptd')