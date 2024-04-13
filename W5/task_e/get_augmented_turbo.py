import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import sys
import json

class Augmented_Turbo(Dataset):
    def __init__(self, root_dir, json_dir):
        self.root_dir = root_dir
        self.json_dir = json_dir
        self.animals = os.listdir(self.root_dir)

        # Get Images
        self.image_paths = []
        self.ids = []
        self.captions = []

        # Create dict of captions
        with open(self.json_dir, 'r') as f:
            self.json_data = json.load(f)

        self.caption_dict = {}
        for item in self.json_data:
            image_id = item['image_id']
            animal = image_id.split('_')[0]
            self.image_paths.append(os.path.join(self.root_dir, animal, image_id + '.png'))
            self.ids.append(item['id'])
            self.captions.append(item['caption'])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        id = self.ids[idx]
        caption = self.captions[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, caption, id


if __name__ == "__main__":
    root_dir = '/ghome/group07/C5-W5/task_d/augmented_turbo_100/'  
    json_dir = '/ghome/group07/C5-W5/task_d/augmented_dataset_turbo_100.json'     
    dataset = Augmented_Turbo(root_dir=root_dir, json_dir=json_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    for img_paths, captions, ids in dataloader:
        print(img_paths.shape)
        print(ids)
        print(captions)
        sys.exit()
