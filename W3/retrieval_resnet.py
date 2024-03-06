import torch
import sys
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import Dataset
from torchvision import models
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import average_precision_score


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
    def __init__(self, embed_size=2096, num_epochs=10, batch_size=8):
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

    
    def forward(self, img):
        with torch.no_grad():
            return self.model(img)
    

    def extract_features(self, dataloader):
        features = []
        targets = []
        for batch_idx, (images, labels, captions) in enumerate(dataloader):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.forward(images)
            features.append(outputs.cpu().detach().numpy())
            targets.append(labels.cpu().numpy())
        return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)
    
    
    def train(self, dataloader):
        features, labels = self.extract_features(dataloader)
        # Train the classifier
        # clf = svm.SVC(kernel='rbf')
        clf = KNeighborsClassifier(n_neighbors=13,n_jobs=-1,metric='euclidean')
        clf.fit(features, labels)
        return clf , labels
    
    
    def test(self, labels_train, dataloader_test, model):
        features_test, labels_test = self.extract_features(dataloader_test)
        # Test
        #acc = model.score(features, labels)
        #print(f"Accuracy Test KNN: {acc}")

        neighbors = model.kneighbors(features_test, return_distance=False)
        neighbors_labels = []
    
        #print(f"features {features.shape} labels {labels.shape} neighbors {neighbors.shape}")
        for i in range(len(neighbors)):
            neighbors_class = []
            for j in neighbors[i]:
                neighbors_class.append((labels_train[j] == labels_test[i])*1)
                #print(labels_train[j], labels_test[i])
            # print(neighbors_class)
            neighbors_labels.append(neighbors_class)
        
        avg = average_precision_score(neighbors_labels,labels_test.reshape(-1, 1))

        print(avg)
        

        
        





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
    clf, train_labels = model.train(train_dataloader)
    model.test(train_labels, test_dataloader, clf)
    



    


