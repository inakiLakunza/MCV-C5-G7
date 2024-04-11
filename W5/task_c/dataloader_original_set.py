
import os
import sys

import random
import tqdm
from PIL import Image
from pathlib import Path

import json
        
import torch
from torchvision import transforms



class OriginalDataset():
    def __init__(self, train=True):
        PATH_PARENT_DIRECTORY = "./../../mcv/datasets/C5/COCO"
        
        self.train = train
        if self.train: 
            self.img_path = os.path.join(PATH_PARENT_DIRECTORY, "train2014")
            self.json_path = os.path.join(PATH_PARENT_DIRECTORY, "captions_train2014.json")
        else:
            self.img_path = os.path.join(PATH_PARENT_DIRECTORY, "val2014")
            self.json_path = os.path.join(PATH_PARENT_DIRECTORY, "captions_val2014.json")

        json_file = open(self.json_path)
        self.info = json.load(json_file)['annotations']
              
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        query = self.info[index]
        anchor_id = query['image_id']
        id = query['id']
        caption = query['caption']

        if self.train: anchor_path = os.path.join(self.img_path, 'COCO_train2014_' + str(anchor_id).zfill(12) + '.jpg')
        else: anchor_path = os.path.join(self.img_path, 'COCO_val2014_' + str(anchor_id).zfill(12) + '.jpg')
        
        anchor_img = self.transform(Image.open(anchor_path).convert('RGB')).to(self.device)

        #return anchor_img, caption, id
        return caption, id

    def get_random_image(self):
        random_index = random.randint(0, len(self.info) - 1)
        return self.__getitem__(random_index)