import torch
import sys
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import Dataset
from torchvision import models
import os
from sklearn.metrics import average_precision_score, top_k_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from metrics import mapk
#from losses import ContrastiveLoss, TripletLoss
import pytorch_metric_learning
from pytorch_metric_learning import trainers
from pytorch_metric_learning import losses, samplers


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
    

class ResNet_embedding(torch.nn.Module):
    def __init__(self, embed_size=32, num_epochs=10, batch_size=8, loss= losses.ContrastiveLoss()):
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
        self.loss = loss
        print("loss:", self.loss)

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
    
    def train_model(self, train_dataloader, val_dataloader, n_epochs=100):

        data_labels = [x for _, x in train_dataloader.samples]
        class_sampler = samplers.MPerClassSampler(
        labels= data_labels,
        m=16 // 8,
        batch_size=16,
        length_before_new_iter=len(train_dataloader),
        )
        print(self.loss)

        metric_trainer = trainers.MetricLossOnly(
            models={"trunk": self},
            optimizers={"trunk_optimizer": self.optimizer},
            batch_size= 64,            #config["batch_size"],
            loss_funcs={"metric_loss": self.loss},
            #mining_funcs=mining_funcs,  # {"subset_batch_miner": mining_func1, "tuple_miner": mining_func2}
            dataset= train_dataloader,
            data_device= self.device,
            sampler=class_sampler, # SI QUEREMOS USAR ALGUN SEED
            #lr_schedulers= {"trunk":self, "step_type" : optim.lr_scheduler.StepLR(self.optimizer, step_size=2,gamma=0.9)},
            lr_schedulers={"trunk_scheduler_by_epoch": optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.9)},
    
            #end_of_iteration_hook=hooks.end_of_iteration_hook, # PARA HACER LO QUE QUERAMOS EN CADA ITERATION, COMO LOGGEAR LOSSES
            #end_of_epoch_hook=end_of_epoch_hook, #PARA HACER VALIDATION, GUARDAR MEJOR MODELO, Y LOGS DURANTE EL ENTRENAMIENTO
        )
        print(train_dataloader)
        metric_trainer.train(1, n_epochs)   

    
    def train_knn(self, dataloader):
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
        #apk1 = top_k_accuracy_score(neighbors_labels, labels_test,k=1)
        #apk5 = top_k_accuracy_score(neighbors_labels, labels_test.reshape(-1, 1),k=5)

        mapk1 = mapk(labels_test.reshape(-1, 1), neighbors_labels, k=1)
        mapk5 = mapk(labels_test.reshape(-1, 1), neighbors_labels, k=5)


        return avg, mapk1, mapk5
    
    

class SiameseNet(torch.nn.Module):    
    def __init__(self, embedding_net):        
        super(SiameseNet, self).__init__()        
        self.embedding_net = embedding_net 

    def forward(self, x1, x2):        
        output1 = self.embedding_net(x1)        
        output2 = self.embedding_net(x2)        
        return output1, output2  
      
    def get_embedding(self, x):        
        return self.embedding_net(x)