import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import argparse

    
def prepare_from_vg_path(path):
    
    # first create folders for each triplet
    triplets = pd.read_csv('simat_db/triplets.csv')
    triplet_names = list(set(f'A {l.subj} {l.rel} {l.obj}' for i, l in triplets.iterrows()))
    Path('simat_db/images').mkdir(exist_ok=True)
    for t in triplet_names:
        Path(f'simat_db/images/{t}').mkdir(exist_ok=True)
        
    # add images in the right folder
    retrieval_db = pd.read_csv('simat_db/retrieval_db.tsv', sep='\t', index_col=0)
    rid2iid = dict(zip(retrieval_db.index, retrieval_db.image_id))
    for i, l in tqdm(triplets.iterrows()):
        img = Image.open(path + str(rid2iid[l.region_id])+'.jpg')
        folder = f'A {l.subj} {l.rel} {l.obj}'
        bbox = [int(x) for x in retrieval_db.loc[l.region_id].bbox.split(',')]
        img.save(f'simat_db/images/{folder}/{l.region_id}.jpg')        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare SIMAT dataset')
    parser.add_argument('--VG_PATH', type=str,
                        help='where the Visual Genome dataset is stored')
    args = parser.parse_args()
    
    prepare_from_vg_path(args.VG_PATH)
    