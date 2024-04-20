
import os
import sys
sys.path.append('./..') 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import pickle
from torchvision.transforms import v2
from img_dataloader import Dataset
import tqdm
import torch
import torch.optim as optim
from torchvision import transforms
from transformers import BertTokenizer, BertModel
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import ConcatDataset, DataLoader
import librosa
import librosa.display
import noisereduce as nr
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1
import sys

from filter_datasets import FilteredImageFolder


class Model():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # MODEL FOR IMAGES
        #==========================================================
        print("Loading BERT...")
        self.model = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=7).to(self.device)
        #==========================================================

        # LOSSES AND OPTIMIZERS
        #==========================================================
        params = self.model.parameters()
        self.image_optimizer = optim.AdamW(params, lr=1e-6, weight_decay=0.)

        N_AUGMENTATIONS: int = 3
        train_data_per_category_augmented: list[int] = [10*N_AUGMENTATIONS, 164*N_AUGMENTATIONS, 1264, 2932, 1353, 232*N_AUGMENTATIONS, 51*N_AUGMENTATIONS]
        n_elements_augmented: int = sum(train_data_per_category_augmented)

        class_weigths_augmented = [num/n_elements_augmented for num in train_data_per_category_augmented]
        class_weights_augmented = torch.tensor(class_weigths_augmented).to(self.device)
        self.image_loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights_augmented).to(self.device))
        #==========================================================

        self.best_acc = 0



    def train(self, dataloader, val_dataloader, n_epochs=1) -> None:
        for epoch in range(n_epochs):
            self.model.train()
            running_loss = []
            running_acc = []
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            for batch_idx, data in pbar:
                images, labels = data

                images = images.to(self.device)
                labels = labels.to(self.device)
                labels = torch.add(labels, -1)

                assert labels.min() >= 0 and labels.max() <= 6, "Label indices out of range"

                self.image_optimizer.zero_grad()
                
                # Predict
                outputs = self.model(images)
                pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

                # Compute Loss
                image_loss = self.image_loss_fn(outputs, labels)
                image_loss.backward()
                self.image_optimizer.step()

                # Compute Accuracy
                correct_predictions = (pred_class == labels).float()
                accuracy = correct_predictions.sum() / len(labels)

                running_loss.append(image_loss.item())
                running_acc.append(accuracy)

            loss, acc = self.test(val_dataloader)
            print(f'EPOCH {epoch} |  Train Loss: {round(torch.Tensor(running_loss).mean().item(), 4)}  Train Acc: {round(torch.Tensor(running_acc).mean().item(), 4)}    Val Loss: {round(loss.item(), 4)}  Val Acc: {round(acc.item(), 4)}')
            
            if acc > self.best_acc:
                save_path_txt = './weights/image_model.pth'
                os.makedirs("./weights", exist_ok=True)
                torch.save(self.model.state_dict(), save_path_txt)
                self.best_acc = acc
        

    def test(self, test_dataloader):
        self.model.eval()

        loss = 0
        accuracy = 0
        batches = 0

        with torch.no_grad():
            for batch_idx, data in enumerate(test_dataloader):
                images, labels = data

                images = images.to(self.device)
                labels = labels.to(self.device)
                labels = torch.add(labels, -1)
                assert labels.min() >= 0 and labels.max() <= 6, "Label indices out of range"

                # pred
                outputs = self.model(images)
                pred_class = torch.argmax(outputs, dim=1)

                # Loss
                loss += self.image_loss_fn(outputs, labels)

                # Acc
                correct_predictions = (pred_class == labels).float()
                accuracy += correct_predictions.sum() / len(labels)
                batches += 1

            average_loss = loss / batches
            average_accuracy = accuracy / batches

        return average_loss, average_accuracy


def _filter_samples(dataset, filter_labels=[3, 4, 5]):
    filtered_samples = []
    for idx, (sample) in enumerate(dataset):
        _, label = sample
        if label in filter_labels:
            continue
        filtered_samples.append(sample)
    return filtered_samples

if __name__ == '__main__':
    batch_size = 256

    model = Model()
    print("Loading data...")
    # DATALOADER =========================================================
    train_dataset = Dataset(regime='train')
    val_dataset = Dataset(regime='val')
    test_dataset = Dataset(regime='test')
    train_dataset_augmented_1 = Dataset(aug='train_aug_1')
    train_dataset_augmented_2 = Dataset(aug='train_aug_2')

    filtered_dataset_1 = _filter_samples(train_dataset_augmented_1)
    filtered_dataset_2 = _filter_samples(train_dataset_augmented_2)

    print(f'Length of filtered_dataset_1 is {len(filtered_dataset_1)}')
    print(f'Length of filtered_dataset_2 is {len(filtered_dataset_2)}')

    train_dataset = ConcatDataset([train_dataset, filtered_dataset_1, filtered_dataset_2])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=2, shuffle=True)

    # ====================================================================
    # Unfreeze layers
    for name, para in model.model.named_parameters():
        para.requires_grad = True
    model.train(train_dataloader, val_dataloader, n_epochs=300)
    print(f'\nFINAL TEST ==============================')
    loss, acc = model.test(test_dataloader)
    print(f'Loss: {loss} Acc: {acc}')


