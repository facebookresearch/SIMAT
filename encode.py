# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import clip
import torch
import torchvision.datasets as datasets
from functools import partial
from tqdm import tqdm
import pandas as pd
from pathlib import Path

# code for encoding the SIMAT database with CLIP
# produces the files data/simat_img_clip_2.pt and data/simat_words_clip_2.ptd

device = 'cuda:1'

DATA_PATH = 'simat_db/images/'
CLIP_MODEL = 'ViT-B/32'

model, prep = clip.load(CLIP_MODEL, device=device)

ds = datasets.ImageFolder(DATA_PATH, transform=prep)

dl = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=10, shuffle=False)

img_enc = torch.cat([model.encode_image(b.to(device)).cpu().detach() for b, i in tqdm(dl)]).float()

fnames = [x[0].name for x in datasets.ImageFolder(DATA_PATH, loader=Path)]
region_ids = [int(x[:-4]) for x in fnames]

img_enc_mapping = dict(zip(region_ids, img_enc))
torch.save(img_enc_mapping, 'data/simat_img_clip_2.pt')

# encode words
transfos = pd.read_csv('simat_db/transfos.csv', index_col=0)
words = list(set(transfos.target) | set(transfos.value))
tokens = clip.tokenize(words)

word_encs = torch.cat([model.encode_text(b.to(device)).cpu().detach() for b in tqdm(tokens.split(32))])

w2we = dict(zip(words, word_encs))
torch.save(w2we, 'data/simat_words_clip_2.ptd')
