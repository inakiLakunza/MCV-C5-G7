import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import sys
import json

class Augmented_2_1(Dataset):
    def __init__(self, root_dir, json_dir):
        self.root_dir = '/ghome/group07/C5-W5/task_d/augmented_2_1/'  
        self.json_dir = '/ghome/group07/C5-W5/task_d/augmented_dataset_2_1.json'      
        self.animals = os.listdir(self.root_dir)

        # Get Images
        self.image_paths = []
        self.ids = []
        for animal in self.animals:
            animal_dir = os.path.join(self.root_dir, animal)
            for img in os.listdir(animal_dir):
                if img.endswith('.png'):
                    self.image_paths.append(os.path.join(animal_dir, img))
                    self.ids.append(img.split('_')[-1].split('.')[0])

        # Create dict of captions
        with open(self.json_dir, 'r') as f:
            self.json_data = json.load(f)

        self.caption_dict = {}
        for item in self.json_data:
            image_id = item['image_id']
            id = image_id.split('_')[-1]
            caption = item['caption']
            self.caption_dict[id] = caption

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        id = self.ids[idx]
        caption = self.caption_dict[id]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, caption, id


if __name__ == "__main__":
    root_dir = '/ghome/group07/C5-W5/task_d/augmented_2_1/'  
    json_dir = '/ghome/group07/C5-W5/task_d/augmented_dataset_2_1.json'     
    dataset = Augmented_2_1(root_dir=root_dir, json_dir=json_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    for img_paths, captions, ids in dataloader:
        print(img_paths.shape)
        print(ids)
        print(captions)
        sys.exit()
