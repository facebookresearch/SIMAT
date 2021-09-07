This is the repository for launching code related to the SIMAT database

see paper arxiv:...

# Download dataset

The SIMAT database is composed of images from COCO

```python prepare_dataset --COCO_PATH=...```

# Perform inference with CLIP

from core. import 

image = Image.open('')
dv = ...
target = img_enc + 


db = torch.load('db_path..')

# optionally, apply linear layer

retrieve target in db
output image

# Compute SIMAT scores with CLIP

python eval.py --mode = clip --lbd [values] --tau=...


# Train adaptation layers on COCO