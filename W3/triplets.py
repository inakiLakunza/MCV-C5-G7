import torch
import sys
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import Dataset
from torchvision import models
import os


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, embed_size):
        super(EmbeddingLayer, self).__init__()
        self.linear = torch.nn.Linear(512, embed_size)
        self.activation = torch.nn.ReLU()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        x = x.to(self.device)
        x = x.squeeze(-1).squeeze(-1)
        x = self.activation(x)
        x = self.linear(x)
        return x
    

class Model(torch.nn.Module):
    def __init__(self, embed_size=2, num_epochs=10, batch_size=8):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Initializing Model with {}\n".format(self.device))

        # Model
        self.model = models.resnet18(pretrained=True, progress=False)
        embed = EmbeddingLayer(embed_size)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1], embed)
        self.model.to(self.device)

        # Params
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.loss = torch.nn.TripletMarginLoss(margin=1.0)

        self.best_val_loss = float('inf')
        self.save_path = './resnet_weights/'
    

    def train(self, train_dataloader, val_dataloader):
        self.model.train()

        for epoch in range(self.num_epochs):
            epoch_accuracy = 0.0
            epoch_loss = 0.0
            for batch_idx, (images, labels, captions) in enumerate(train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images) # returns the prediction logits 
                print(outputs.shape)

                batch_loss = 0.0
                for i in range(len(images)):
                    actual = outputs[i] 
                    indices_for_close = np.squeeze(np.where(labels.cpu() == labels[i].cpu()))
                    indices_for_far = np.squeeze(np.where(labels.cpu() != labels[i].cpu()))
                    if indices_for_close.size == 0 or indices_for_far == 0:
                        continue
                    close = outputs[indices_for_close] # queremos acercar
                    far = outputs[indices_for_far] # take the last far

                    print(labels)

                    print(f"actual.shape: {actual.shape} close.shape {close.shape} far.shape {far.shape}")
                    batch_loss += self.loss(actual, close, far) # triplet loss

                batch_loss /= len(images)
                print(batch_loss)

                # Triplet
                print(labels)
                sys.exit()
                # []






if __name__ == '__main__':
    batch_size = 8
    num_epochs = 15
    weigths_path = None

    print("Loading data.")
    MIT_split_train = Dataset('/ghome/group07/mcv/datasets/C3/MIT_split/train')
    MIT_split_test = Dataset('/ghome/group07/mcv/datasets/C3/MIT_split/test')

    # Split training into Train - Val
    total_size = len(MIT_split_train)
    val_size = int(total_size * 0.3)
    train_size = total_size - val_size
    MIT_split_train, MIT_split_val = random_split(MIT_split_train, [train_size, val_size])

    train_dataloader = DataLoader(MIT_split_train, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(MIT_split_val, batch_size=16, shuffle=True)

    print("Looping through test dataloader...")
    test_dataloader = DataLoader(MIT_split_test, batch_size=16, shuffle=True)
    for batch_idx, (images, labels, captions) in enumerate(test_dataloader):
        print(batch_idx, images.shape, labels, captions)

    print("\nInitializing the Model...")
    model = Model()
    model.train(train_dataloader, val_dataloader)

    


