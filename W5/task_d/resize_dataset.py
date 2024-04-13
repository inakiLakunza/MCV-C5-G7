import sys
import os
import tqdm
import json
import torch
from torch.utils.data import DataLoader
from dataloader_original_animal import OriginalAnimalDataset
from PIL import Image

if __name__ == '__main__':
    PATH_PARENT_DIRECTORY = "./../../mcv/datasets/C5/COCO"

    img_path = os.path.join(PATH_PARENT_DIRECTORY, "train2014")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"The device we are working on is: {DEVICE}")

    animal_subset = OriginalAnimalDataset(True)
    print("Animal set created")

    dataloader_animal_subset = DataLoader(animal_subset,
                                          batch_size=16,
                                          drop_last=False,
                                          )
    
    for j, element in enumerate(dataloader_animal_subset):
        anchor_ids, ids, captions = element
        for i in range(len(anchor_ids)):
            save_path = './resized_dataset/'
            anchor_path = os.path.join(img_path, 'COCO_train2014_' + str(anchor_ids[i].item()).zfill(12) + '.jpg')
            image = Image.open(anchor_path)
            if 'horse' in captions[i].lower().split():
                save_path = os.path.join(save_path, 'horse')
            elif 'zebra' in captions[i].lower().split():
                save_path = os.path.join(save_path, 'zebra')
            elif 'cow' in captions[i].lower().split():
                save_path = os.path.join(save_path, 'cow')
            elif 'elephant' in captions[i].lower().split():
                save_path = os.path.join(save_path, 'elephant')
            elif 'sheep' in captions[i].lower().split():
                save_path = os.path.join(save_path, 'sheep')
            elif 'giraffe' in captions[i].lower().split():
                save_path = os.path.join(save_path, 'giraffe')
            else:
                print(captions[i])
                print("No lee ningun animal!")
                continue
            os.makedirs(save_path, exist_ok=True)
            name_file = str(i + j).zfill(4) + '.png'
            save_path = os.path.join(save_path, name_file)
            resized_image = image.resize((224, 224), Image.Resampling.LANCZOS)
            resized_image.save(save_path)
