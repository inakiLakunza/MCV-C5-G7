import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
import random

class Dataset():
    def __init__(self, data_path: str):
        # data_path for MIT_split should be MIT_split/train or MIT_split/test
        self.data_path = data_path
        self.images = []
        self.labels = []
        self.captions = []

        self.categories = os.listdir(self.data_path)
        ctr = 0
        for cat in sorted(self.categories):
            regexp = os.path.join(self.data_path, cat, '*.jpg')
            imgs = sorted(glob.glob(regexp))
            for img in imgs:
                self.images.append(img)
                self.labels.append(ctr)
                self.captions.append(cat)
            ctr += 1
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        label = self.labels[index]
        caption = self.captions[index]
        return img_tensor, torch.tensor(label), caption

        


