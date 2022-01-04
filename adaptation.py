# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import sys
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import argparse
import torchvision.datasets as datasets
from functools import partial
import torch
import clip 

COCO_ROOT_TRAIN = '/checkpoint/gcouairon/coco/full'
COCO_ANN_TRAIN = '/checkpoint/gcouairon/coco/train_annotations.json'
COCO_ROOT_VAL = '/checkpoint/gcouairon/coco/full'
COCO_ANN_VAL = '/checkpoint/gcouairon/coco/val_annotations.json'
    
def main():
    parser = argparse.ArgumentParser(description='Train adaptation layers')
    parser.add_argument('--backbone', type=str, default='ViT-B/32', help='which CLIP model to use')
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--output_dim', type=int, default=512)
    parser.add_argument('--tau', type=float, default=0.1)
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sched_step_size', type=int, default=25)
    parser.add_argument('--sched_gamma', type=float, default=0.1)
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    
    
    args = parser.parse_args()
    train_adaptation_layers(args)
    

def clip_loss(imgf, txtf, T = 0.01):
    imgf = imgf / imgf.norm(2, dim=-1, keepdim=True)
    txtf = txtf / txtf.norm(2, dim=-1, keepdim=True)
        
    mma = (imgf @ txtf.T)/T # mma for MultiModalAlignment

    labels = torch.arange(mma.shape[0], device=mma.device)

    loss1 = F.cross_entropy(mma, labels)
    loss2 = F.cross_entropy(mma.T, labels)
    loss = (loss1 + loss2)/2
    
    return loss
  


class MMEncoder(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.core, _ = clip.load(args.backbone, device='cpu', jit=False)
        
        self.img_head = nn.Linear(512, args.output_dim)
        self.init_head(self.img_head)
        
        self.txt_head = nn.Linear(512, args.output_dim)
        self.init_head(self.txt_head)
        
        
    def init_head(self, x, eps=0.01):
        x.weight.data = torch.eye(*x.weight.data.shape) + eps * x.weight.data
        x.bias.data = eps * x.bias.data
    
    
    def training_step(self, batch, batch_nb):
        img, txt = batch
        with torch.no_grad():
            img_ = self.core.encode_image(img).detach()
            txt_ = self.core.encode_text(txt).detach()
        
        img_ = self.img_head(img_)
        txt_ = self.txt_head(img_)
        
        loss = clip_loss(img_, txt_, T=self.T)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_nb):
        img, txt = batch
        with torch.no_grad():
            img_ = self.core.encode_image(img).detach()
            txt_ = self.core.encode_text(txt).detach()
        
        img_ = self.img_head(img_)
        txt_ = self.txt_head(img_)
        
        loss = clip_loss(img_, txt_, T=self.T)
                
        self.log('val_loss', loss)
        return img_, txt_
        
    
    def configure_optimizers(self):
        opt = torch.optim.Adam([
                {'params': self.img_head.parameters(), 'lr': self.args.lr},
                {'params': self.txt_head.parameters(), 'lr': self.args.lr}
            ], lr=self.args.lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size = self.args.sched_step_size, gamma = self.args.sched_gamma)
        return [opt], [sched]

def train_adaptation_layers(args):
    
    transform = T.Compose([T.Resize(size=256, interpolation=Image.BICUBIC),
                                T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
                                T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                        std=(0.26862954, 0.26130258, 0.27577711))])
    
    train_dataset = datasets.CocoCaptions(root = COCO_ROOT_TRAIN,
                        annFile = COCO_ANN_TRAIN, transform=transform)
    
    val_dataset = datasets.CocoCaptions(root = COCO_ROOT_VAL,
                        annFile = COCO_ANN_VAL, transform=transform)
    
    collate_fn = lambda x:(torch.cat([xi[0] for xi in x]), clip.tokenize([xi[1][np.random.randint(5)] for xi in x]))
    
    dl = partial(torch.utils.data.DataLoader, 
                 batch_size=args.batch_size,
                 num_workers=10,
                 collate_fn = collate_fn)
                                    
    
    train_dataloader = dl(train_dataset, shuffle=True)
    
    val_dataloader = dl(val_dataset, shuffle=False)
    
    model = MMEncoder(args)
    
    trainer = pl.Trainer(precision=args.precision,
            gpus=args.gpus,
            num_nodes=args.nodes,
            gradient_clip_val=1,
            accelerator='ddp',
            max_epochs=args.max_epochs,
            progress_bar_refresh_rate=10)
            
    trainer.fit(model, train_dataloader, [val_dataloader])
    
    
    
if __name__ == '__main__':
    main()
