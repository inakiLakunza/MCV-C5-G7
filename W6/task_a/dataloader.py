from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm

def parse_data(data):
    return [{"VideoName":  record.split(',')[0],
              "UserID":    record.split(',')[1], 
              "AgeGroup":  record.split(',')[2], 
              "Gender":    record.split(',')[3], 
              "Ethnicity": record.split(',')[4]} for record in tqdm(data[1:])]

class Dataset():
    def __init__(self, regime="train"):

        assert regime.lower() in ["train", "val", "test"], "Chosen regime is not correct"

        if regime == 'train':
            LABELS_PATH = '/ghome/group07/C5-W6/First_Impressions_v3_multimodal/train_set_age_labels.csv'
        elif regime == 'val':
            LABELS_PATH = '/ghome/group07/C5-W6/First_Impressions_v3_multimodal/valid_set_age_labels.csv'
        else:
            LABELS_PATH = '/ghome/group07/C5-W6/First_Impressions_v3_multimodal/test_set_age_labels.csv'

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(LABELS_PATH, 'r', encoding="utf-8") as f:
            DATA = f.read().splitlines()

        self.data = parse_data(DATA)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record = self.data[index]
        return record["AgeGroup"], record["Gender"], record["Ethnicity"]