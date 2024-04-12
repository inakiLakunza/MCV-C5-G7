


import os
import sys

import random
import tqdm
from PIL import Image
from pathlib import Path

import json
        
import torch
from torchvision import transforms



class OriginalAnimalDataset():
    def __init__(self, train=True):
        PATH_PARENT_DIRECTORY = "./../../mcv/datasets/C5/COCO"
        self.horse_subset = []
        self.zebra_subset = []
        self.cow_subset = []
        self.elephant_subset = []
        self.sheep_subset = []
        self.giraffe_subset = []
        
        self.train = train
        if self.train: 
            self.img_path = os.path.join(PATH_PARENT_DIRECTORY, "train2014")
            # WE USE THE CREATED JSON, JUST FOCUSING ON THE ANIMALS
            self.json_path = os.path.join("animal_dataset.json")
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            for element in data:
                if element.get('animal') == 'horse':
                    self.horse_subset.append(element)
                elif element.get('animal') == 'zebra':
                    self.zebra_subset.append(element)
                elif element.get('animal') == 'cow':
                    self.cow_subset.append(element)
                elif element.get('animal') == 'elephant':
                    self.elephant_subset.append(element)
                elif element.get('animal') == 'sheep':
                    self.sheep_subset.append(element)
                elif element.get('animal') == 'giraffe':
                    self.giraffe_subset.append(element)
        else:
            self.img_path = os.path.join(PATH_PARENT_DIRECTORY, "val2014")
            self.json_path = os.path.join(PATH_PARENT_DIRECTORY, "captions_val2014.json")

        json_file = open(self.json_path)
        self.info = json.load(json_file)

        # THE JSON WE HAVE SAVED DOES NOT HAVE AN ANNOTATIONS PART,
        # WE GET IT DIRECTLY, BUT IN THE ORIGINAL DATASETS, WE HAVE 
        # TO RETRIEVE THE 'annotations' KEY'S VALUE
        if not train:
            self.info = self.info['annotations']

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        # ESTO TAMBIEN HABRIA QUE CAMBIARLO CON LO DEL GETITEM
        # SI VAMOS A USAR ESTE DATASET PARA HACER FINE TUNING
        return len(self.info)

    def __getitem__(self, index):

        # !!! HAY QUE CAMBIAR ESTO, SI VAMOS A ENTRENAR O HACER FINE TUNING
        # SOLO CON LOS ANIMALES HAY QUE HACER QUE PILLE UN ANIMAL RANDOM
        # DE LOS QUE HAY Y LUEGO HACER EL INDEX DENTRO, PERO TAMBIEN HAY 
        # QUE CAMBIAR LO DEL INDEX PORQUE HAY QUE PONER CUANTOS TENEMOS
        # EN TOTAL, PERO NOSOTROS LO TENEMOS POR SUBSECCIONES, SINO
        # ES HACER UNA SUMA CON TODAS LAS QUE HAY, EN PLAN
        # {HORSE: 7000, SHEEP: 2000, ...}, Y EL TOTAL POR EJEMPLO 40000
        # ENTONCES SI BUSCAMOS EL IDX=8000, POR EJEMPLO, QUE MIRE DONDE
        # DEBERIA CAER, EN ESTE CASO EN SHEEP PORQUE 7000<8000, Y QUE PILLE
        # LA 8000-7000=1000 DE SHEEP  

        query = self.info[index]
        anchor_id = query['image_id']
        id = query['id']
        caption = query['caption']

        if self.train: anchor_path = os.path.join(self.img_path, 'COCO_train2014_' + str(anchor_id).zfill(12) + '.jpg')
        else: anchor_path = os.path.join(self.img_path, 'COCO_val2014_' + str(anchor_id).zfill(12) + '.jpg')
        
        anchor_img = self.transform(Image.open(anchor_path).convert('RGB')).to(self.device)

        # CHOOSE WANTED RETURN
        #==================================
        #return anchor_img, caption, id
        #return caption, id
        return anchor_id, id, caption
        #==================================

    def get_random_image(self):
        random_index = random.randint(0, len(self.info) - 1)
        return self.__getitem__(random_index)
    
    def get_wanted_random_caption(self, animal):
        animal_list = self.info[animal]
        random_index = random.randint(0, len(animal_list)-1)
        return animal_list[random_index]["caption"]

    def get_wanted_random_caption_of_subset(self, subset):
        random_index = random.randint(0, len(subset)-1)
        return subset[random_index]
