import torch
import sys
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import Dataset
from torchvision import models
import os

class Model:
    def __init__(self, batch_size, num_epochs, weights=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Initializing Model with {}\n".format(self.device))

        # Model
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 8) # Replace the last layer for our number of classes
        self.model = self.model.to(self.device)

        # Params
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.loss = torch.nn.CrossEntropyLoss()

        self.best_val_loss = float('inf')
        self.save_path = './resnet_weights/'

        if weights is not None:
            self.model.load_state_dict(torch.load(weights))

    
    def validate(self, val_dataloader):
        self.model.eval() # turn eval mode to avoid updating
        with torch.no_grad():
            validation_accuracy = 0.0
            validation_loss = 0.0
            for batch_idx, (images, labels, captions) in enumerate(val_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images) # returns the prediction logits 

                # Cross-Entropy loss
                loss = self.loss(outputs, labels)
                validation_loss += loss.item()

                # Compute Accuracy
                preds = torch.argmax(outputs, dim=1) # get idx of max logit
                correct = (preds == labels)
                validation_accuracy += (torch.sum(correct)).item()
            self.model.train()
            val_acc = validation_accuracy / len(val_dataloader.dataset)
            val_loss = validation_loss / len(val_dataloader.dataset)
        return val_acc, val_loss
    

    def train(self, train_dataloader, val_dataloader):
        self.model.train()

        for epoch in range(self.num_epochs):
            epoch_accuracy = 0.0
            epoch_loss = 0.0
            for batch_idx, (images, labels, captions) in enumerate(train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images) # returns the prediction logits 

                # Cross-Entropy loss
                loss = self.loss(outputs, labels)
                epoch_loss += loss.item()

                # Compute Accuracy
                preds = torch.argmax(outputs, dim=1) # get idx of max logit
                correct = (preds == labels)
                epoch_accuracy += (torch.sum(correct)).item()

                # Backprop and update optimizer
                loss.backward()
                self.optimizer.step()
            
            train_acc = epoch_accuracy / len(train_dataloader.dataset)
            train_loss = epoch_loss / len(train_dataloader.dataset)
            
            val_acc, val_loss = self.validate(val_dataloader)

            print(f"Epoch [{str(epoch + 1).zfill(3)}/{str(self.num_epochs).zfill(3)}]  Train Accuracy: {train_acc:.4f}  Val Accuracy: {val_acc:.4f}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

            if val_loss <= self.best_val_loss:
                #print(f"Saving model with best validation accuracy... Val acc: {val_acc:.4f}")
                os.makedirs(self.save_path, exist_ok=True)
                filename = os.path.join(self.save_path, 'checkpoint.pth')
                torch.save(self.model.state_dict(), filename)
                self.best_val_loss = val_loss


    def test(self, test_dataloader, weight_path=None):
        if weight_path is None:
            weight_path = os.path.join(self.save_path, 'checkpoint.pth')
        
        print(f"\n\nLoading model for testing from {weight_path}...")
        self.model.load_state_dict(torch.load(weight_path))
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            test_accuracy = 0.0
            test_loss = 0.0
            for batch_idx, (images, labels, captions) in enumerate(test_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images) # returns the prediction logits 

                # Cross-Entropy loss
                loss = self.loss(outputs, labels)
                test_loss += loss.item()

                # Compute Accuracy
                preds = torch.argmax(outputs, dim=1) # get idx of max logit
                correct = (preds == labels)
                test_accuracy += (torch.sum(correct)).item()
        return test_accuracy / len(test_dataloader.dataset), test_loss / len(test_dataloader.dataset)
        
        


if __name__ == '__main__':
    batch_size = 16
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
    model = Model(batch_size, num_epochs, weights=weigths_path)
    model.train(train_dataloader, val_dataloader)
    acc, loss = model.test(test_dataloader)

    print(f"[TEST] Accuracy: {acc:.4f} Loss: {loss:.4f}")

    


