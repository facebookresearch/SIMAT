# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='prepare SIMAT dataset')
parser.add_argument('--path', type=str,
                    help='where the Visual Genome dataset is stored')
args = parser.parse_args()
path = args.path
    
triplets = pd.read_csv('simat_db/triplets.csv')

Path('simat_db/images').mkdir(exist_ok=True)
Path('simat_db/images/images').mkdir(exist_ok=True)

# add images in the right folder
retrieval_db = pd.read_csv('simat_db/retrieval_db.tsv', sep='\t', index_col=0)
rid2iid = dict(zip(retrieval_db.index, retrieval_db.image_id))
for i, l in tqdm(triplets.iterrows()):
    img = Image.open(path + str(rid2iid[l.region_id])+'.jpg')
    bbox = [int(x) for x in retrieval_db.loc[l.region_id].bbox.split(',')]   
    img.crop(bbox).save(f'simat_db/images/images/{l.region_id}.png')        

    
