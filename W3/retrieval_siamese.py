import torch
import sys
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import average_precision_score
from model_classes import ResNet_embedding
from losses import ContrastiveLoss


        

        




if __name__ == '__main__':
    batch_size = 8
    num_epochs = 15
    weigths_path = None

    print("Loading data.")
    MIT_split_train = Dataset('/ghome/group07/mcv/datasets/C3/MIT_split/train')
    MIT_split_test = Dataset('/ghome/group07/mcv/datasets/C3/MIT_split/test')
     
    dataset = ImageFolder('/ghome/group07/mcv/datasets/C3/MIT_split/train', transform =transforms.ToTensor())
   
                          
    # Split training into Train - Val
    total_size = len(MIT_split_train)
    val_size = int(total_size * 0.3)
    train_size = total_size - val_size
    MIT_split_train, MIT_split_val = random_split(MIT_split_train, [train_size, val_size])

    train_dataloader = DataLoader(MIT_split_train, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(MIT_split_val, batch_size=16, shuffle=True)

    #dataset_train = ImageFolder(MIT_split_train, transform=transforms.ToTensor())
    #dataset_test = ImageFolder(MIT_split_train, transform=transforms.ToTensor())

    print("Looping through test dataloader...")
    test_dataloader = DataLoader(MIT_split_test, batch_size=16, shuffle=True)
    for batch_idx, (images, labels, captions) in enumerate(test_dataloader):
        print(batch_idx, images.shape, labels, captions)

    # SIAMESE NET, SO CONTRASTIVE LOSS
    loss = ContrastiveLoss(margin=1.0)

    print("\nInitializing the Model...")
    model = ResNet_embedding()
    clf, train_labels = model.train_model(dataset, val_dataloader)
    avg_precision, mapk1, mapk5 = model.test(train_labels, test_dataloader, clf)
    print(f"\n\nObtained average precision: {avg_precision}")
    #print(f"\nObtained apk1: {apk1} and apk5: {apk5}")
    print(f"\nObtained mapk1: {mapk1} and mapk5: {mapk5}")



    


