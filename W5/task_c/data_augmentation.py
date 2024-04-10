

import sys
import os

import torch
from torch.utils.data import DataLoader

import numpy as np
import json
from tqdm import tqdm

from dataloader_original_set import OriginalDataset

from keyphrasetransformer import KeyPhraseTransformer





if __name__ == '__main__':
    '''
    print("Loading original dataset...")
    train_set = OriginalDataset(train=True)
    val_set   = OriginalDataset(train=False)

    dataloader_train = DataLoader(train_set, 
                                  batch_size=16,
                                  drop_last=False,
                                  shuffle=False)
    
    dataloader_val   = DataLoader(val_set, 
                                  batch_size=16,
                                  drop_last=False,
                                  shuffle=False)
    

    # ITERATING THROUGH THE DATALOADER GIVES US THE FOLLOWING OUTPUT:
    # ANCHOR_IMGS (16, 225, 225, 3)  , CAPTIONS (16) , IDS (16)
    #anchor_imgs, captions, ids = next(iter(dataloader_val))
    
    #for batch_idx, (anchor_imgs, captions, ids) in dataloader_train:
    '''


    



    

