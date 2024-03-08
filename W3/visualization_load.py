import torch
import sys
import numpy as np
import json

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
#from losses import ContrastiveLoss
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from pytorch_metric_learning import losses
import pandas as pd
import umap


distinct_colors = [
        [r / 255, g / 255, b / 255]
        for r, g, b in [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [128, 0, 128],  # Purple
            [255, 165, 0]   # Orange
        ]
    ]


def display_UMAP_plot(features_x, features_y, captions, title="UMAP", save_path=".", added_name=""):

    features_data = np.hstack([features_x, features_y[:, None]])

    N_COMPONENTS = 2
    out_tsne = umap.UMAP(n_components=N_COMPONENTS, min_dist=0.1, metric='euclidean').fit_transform(features_data)

    df = pd.DataFrame(dict(x=out_tsne[:, 0], y=out_tsne[:, 1], label=captions))
    sns.set_style("whitegrid")
    sns.scatterplot(x="x", y="y", hue="label", palette=distinct_colors, data=df, legend=True)
    plt.title(title)
    if added_name == "":
        save_name = os.path.join(save_path, title.replace(" ", "_")+'.png')
    else:
        save_name = os.path.join(save_path, title.replace(" ", "_")+'_'+added_name+'.png')
    plt.savefig(save_name, dpi=300)


def display_tsne_plot(features_x, features_y, captions, title="TSNE", save_path=".", added_name=""):

    features_data = np.hstack([features_x, features_y[:, None]])
    feature_colors = [distinct_colors[l] for l in features_y]

    N_COMPONENTS = 2
    out_tsne = TSNE(n_components=N_COMPONENTS, verbose=1, metric='euclidean').fit_transform(features_data)

    df = pd.DataFrame(dict(x=out_tsne[:, 0], y=out_tsne[:, 1], label=captions))
    sns.set_style("whitegrid")
    sns.scatterplot(x="x", y="y", hue="label", palette=distinct_colors, data=df, legend=True)
    plt.title(title)
    if added_name == "":
        save_name = os.path.join(save_path, title.replace(" ", "_")+'.png')
    else:
        save_name = os.path.join(save_path, title.replace(" ", "_")+'_'+added_name+'.png')
    plt.savefig(save_name, dpi=300)

if __name__ == '__main__':
    
    f = open('./configs/visualization.json')
    config = json.load(f)

    MODEL_NAME = config["MODEL_NAME"]
    TRAIN_PATH = config["TRAIN_PATH"]
    TEST_PATH = config["TEST_PATH"]
    TASK = config["TASK"].lower()
    SAVE_FOLDER = config["SAVE_FOLDER"]
    ADDED_SAVE_NAME = config["ADDED_SAVE_NAME"]
    

    print("Loading data.")
    MIT_split_train = Dataset(TRAIN_PATH)
    MIT_split_test = Dataset(TEST_PATH) 

    train_dataloader = DataLoader(MIT_split_train, batch_size=16, shuffle=True)
   
    print("Looping through test dataloader...")
    test_dataloader = DataLoader(MIT_split_test, batch_size=16, shuffle=True)
    #for batch_idx, (images, labels, captions) in enumerate(test_dataloader):
    #    print(batch_idx, images.shape, labels, captions)

    

    WEIGHTS_PATH = "/ghome/group07/C5-W3/saved_weights"
    WEIGHTS_PATH = os.path.join(WEIGHTS_PATH, "task_"+TASK, MODEL_NAME+".pt")
    

    load_model = ResNet_embedding()
    load_model.load_state_dict(torch.load(WEIGHTS_PATH))
    
    
    DATA_PATH = "/ghome/group07/C5-W3/results"
    DATA_PATH = os.path.join(DATA_PATH, "task_"+TASK+".json")
    with open(DATA_PATH, 'r') as json_file:
        data = json.load(json_file)

    # Iterate over each dictionary in the JSON data
    for item in data:
        # Check if the 'weights_name' key's value is 'b'
        if item.get('weights_name') == MODEL_NAME:
            # Print or use the dictionary containing the 'weights_name' as 'b'
            model_data = item

    print("THE INFORMATION OF THE SELECTED MODEL IS THE FOLLOWING:\n\n")
    print("TASK:     ", TASK, "\n")
    for k,v in model_data.items():
        print(f"{k} :  {v}")
    print("\n\n")

    features_x, features_y, captions = load_model.extract_features_with_captions(test_dataloader)
    
    if TASK == "b":
        extra = " Siamese"
    else:
        extra = " Triplet"
    
    display_tsne_plot(features_x, features_y, captions, title="TSNE"+extra, save_path=SAVE_FOLDER+'/TSNE', added_name=ADDED_SAVE_NAME)
    display_UMAP_plot(features_x, features_y, captions, title="UMAP"+extra, save_path=SAVE_FOLDER+'/UMAP', added_name=ADDED_SAVE_NAME)

    


