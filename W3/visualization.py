import torch
import sys
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import average_precision_score
from model_classes import ResNet_embedding
from losses import ContrastiveLoss

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


        

        




if __name__ == '__main__':
    batch_size = 8
    num_epochs = 15
    weigths_path = None

    EMB_SHAPE = 2096

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


    # FOR DISPLAYING EACH LABEL
    distinct_colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [128, 0, 128],  # Purple
        [255, 165, 0]   # Orange
    ]

    features_x, features_y = model.extract_features(test_dataloader)
    print(f"""Shape of features_x: {features_x.shape}\n
          Shape of features_ {features_y.shape}""")

    features_data  = np.empty((len(features_y), EMB_SHAPE))
    features_color = np.empty((len(features_y), EMB_SHAPE))
    for i, (x, y) in enumerate(features_x, features_y):
        features_data[i] = (x,y)
        features_color[i] = distinct_colors[y]

    N_COMPONENTS = 2
    out_tsne = TSNE(n_components=N_COMPONENTS, verbose=1, metric='manhattan').fit_transform(
        features_data)

    if N_COMPONENTS == 2:
        scatter_plot = plt.scatter(out_tsne[:, 0], out_tsne[:, 1], c=features_color)
    

    plt.title('TSNE')
    plt.savefig("./tsne_siamese.png")
    print('DONE')

    


