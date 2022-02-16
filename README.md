This repository contains the database and code used in the paper [Embedding Arithmetic for Text-driven Image Transformation](https://arxiv.org/abs/2112.03162) (Guillaume Couairon, Holger Schwenk, Matthijs Douze,  Matthieu Cord)

The inspiration for this work are the geometric properties of word embeddings, such as Queen ~ Woman + (King - Man).
We extend this idea to multimodal embedding spaces (like CLIP), which let us semantically edit images via "delta vectors". 

Transformed images can then be retrieved in a dataset of images.

<p align="center">
    <img src="assets/method.png">
</p>

## The SIMAT Dataset

We build SIMAT, a dataset to evaluate the task of text-driven image transformation, for simple images that can be characterized by a single subject-relation-object annotation. 
A **transformation query** is a pair (*image*, *query*) where the query asks to change the subject, the relation or the object in the input *image*.
SIMAT contains ~6k images and an average of 3 transformation queries per image.

The goal is to retrieve an image in the dataset that corresponds to the query specifications.
We use [OSCAR](https://github.com/microsoft/Oscar) as an oracle to check whether retrieved images are correct with respect to the expected modifications. 



## Examples

Below are a few examples that are in the dataset, and images that were retrieved for our best-performing algorithm.

<p align="center">
    <img src="assets/examples_pos.png" style='width:66%'>
</p>

## Download dataset

The SIMAT database is composed of crops of images from Visual Genome. You first need to install Visual Genome and then run the following command :

```python
python prepare_dataset.py --VG_PATH=/path/to/visual/genome
```

## Perform inference with CLIP ViT-B/32

In this example, we use the CLIP ViT-B/32 model to edit an image. Note that the dataset of clip embeddings is pre-computed.

```python
import clip
from torchvision import datasets
from PIL import Image
from IPython.display import display

#hack to normalize tensors easily
torch.Tensor.normalize = lambda x:x/x.norm(dim=-1, keepdim=True)

# database to perform the retrieval step
dataset = datasets.ImageFolder('simat_db/images/')

db = torch.load('data/simat_img_clip.pt')
db_stacked = torch.stack(list(db.values())).float()

idx2rid = list(db.keys())

model, prep = clip.load('ViT-B/32', device=device)

image = Image.open('simat_db/images/images/98316.png')
img_enc = model.encode_image(prep(image).unsqueeze(0).to('cuda:0')).float().cpu().detach()

txt = ['cat', 'dog']
txt_enc = model.encode_text(clip.tokenize(txt).to('cuda:0')).float().cpu().detach()

# optionally, we can apply a linear layer on top of the embeddings
heads = torch.load(f'data/head_clip_t=0.1.pt')
img_enc = heads['img_head'](img_enc)
txt_enc = heads['txt_head'](txt_enc)

db = heads['img_head'](db).normalize()


# now we perform the transformation step
lbd = 1
target_enc = img_enc.normalize() + lbd * (txt_enc[1].normalize() - txt_enc[0].normalize())


retrieved_idx = (db_stacked @ target_enc.float().T).argmax(0).item()

retrieved_rid = idx2rid[retrieved_idx]

display(Image.open(f'simat_db/images/images/{retrieved_rid}.png'))

```


## Compute SIMAT scores with CLIP

You can run the evaluation script with the following command:

```python
python eval.py --backbone clip --domain dev --tau 0.01 --lbd 1 2
```
It automatically load the adaptation layer relative to the value of *tau*.

## Train adaptation layers on COCO

In this part, you can train linear layers after the CLIP encoder on the COCO dataset, to get a better alignment. Here is an example :

```python
python adaptation.py --backbone ViT-B/32 --lr 0.001 --tau 0.1 --batch_size 512
```

## Citation

If you find this paper or dataset useful for your research, please use the following.
```
@article{gco1embedding,
  title={Embedding Arithmetic for text-driven Image Transformation},
  author={Guillaume Couairon, Matthieu Cord, Matthijs Douze, Holger Schwenk},
  journal={arXiv preprint arXiv:2112.03162},
  year={2021}
}
```

## References

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), OpenAI 2021

Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A. Shamma, Michael S. Bernstein, Fei-Fei Li. [Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations](https://arxiv.org/abs/1602.07332), IJCV 2017

Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, Yejin Choi, Jianfeng Gao, [Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks](https://arxiv.org/abs/1602.07332), ECCV 2020

## License

The SIMAT is released under the MIT license. See LICENSE for details.
