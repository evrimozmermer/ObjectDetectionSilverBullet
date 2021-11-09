# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 20:19:52 2021

@author: tekin.evrim.ozmermer
"""
import torch 
from .augmentations import TransformTrain, TransformEvaluate
from .base import Set
import numpy as np

def load(cfg, val = False):
    if not val:
        ds_tr = Set(cfg.data_root,
                    cfg.dataset,
                    transform = TransformTrain(cfg),
                    label_map = {"o1": 0, "o2": 1, "o3": 2, "o4": 3})
        
        dl_tr = torch.utils.data.DataLoader(ds_tr,
                                            batch_size = cfg.batch_size,
                                            collate_fn = collater,
                                            shuffle = True,
                                            num_workers = 0,
                                            drop_last = True,
                                            pin_memory = True)
        return dl_tr
    else:
        ds_tr = Set(cfg.data_root,
                    cfg.dataset,
                    transform = TransformTrain(cfg),
                    label_map = {"o1": 0, "o2": 1, "o3": 2, "o4": 3})
        
        dl_tr = torch.utils.data.DataLoader(ds_tr,
                                            batch_size = cfg.batch_size,
                                            collate_fn = collater,
                                            shuffle = True,
                                            num_workers = 0,
                                            drop_last = True,
                                            pin_memory = True)
        
        ds_ev = Set(cfg.data_root,
                    cfg.eval,
                    transform = TransformEvaluate(cfg),
                    label_map = {"o1": 0, "o2": 1, "o3": 2, "o4": 3})
        
        dl_ev = torch.utils.data.DataLoader(ds_ev,
                                            batch_size = 1,
                                            shuffle = False,
                                            num_workers = 0,
                                            drop_last = True,
                                            pin_memory = True)
        return dl_tr, dl_ev
    
def collater(data):

    imgs = [s['image'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
        
    rows = [int(s.shape[1]) for s in imgs]
    cols = [int(s.shape[2]) for s in imgs]
    batch_size = len(imgs)

    max_row = np.array(rows).max()
    max_col = np.array(cols).max()

    padded_imgs = torch.zeros(batch_size, 3, max_row, max_col)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :, :int(img.shape[1]), :int(img.shape[2])] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 6)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = torch.from_numpy(annot).float()
    else:
        annot_padded = torch.ones((len(annots), 1, 6)) * -1

    return {'image': padded_imgs, 'annot': annot_padded, 'scale': scales}