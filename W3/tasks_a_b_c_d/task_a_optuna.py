import torch
import os
import sys
import numpy as np
import json
import uuid
from time import time
import optuna

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
from pathlib import Path

from utils import transformstojson





if __name__ == '__main__':

    f = open('./configs/conf_a_b_c.json')
    config = json.load(f)

    #BATCH_SIZE = config["BATCH_SIZE"]
    NUM_OF_EPOCHS = config["NUM_OF_EPOCHS"]
    VAL_RELATIVE_SIZE = config["VAL_RELATIVE_SIZE"]
    EMBED_SIZE = config["EMBED_SIZE"]
    BASE_LR = config["BASE_LR"]
    TRAIN_PATH = config["TRAIN_PATH"]
    TEST_PATH = config["TEST_PATH"]

    SAVE_WEIGHTS_PATH = config["SAVE_WEIGHTS_PATH"]
    SAVE_RESULTS_PATH = config["SAVE_RESULTS_PATH"]


    print("Loading data.")
    MIT_split_train = Dataset(TRAIN_PATH)
    MIT_split_test = Dataset(TEST_PATH)
    
    # DEFINE WANTED PREPROCESSING TRANSFORMATIONS
    preproc_list = []
    # LATER FOR SAVING IN RESULTS:
    save_transforms = transformstojson(preproc_list)

    preprocess = transforms.Compose(preproc_list) 
    #dataset = ImageFolder(TRAIN_PATH, transform =transforms.ToTensor())
   
    # Split training into Train - Val
    total_size = len(MIT_split_train)
    val_size = int(total_size * float(VAL_RELATIVE_SIZE))
    train_size = total_size - val_size
    MIT_split_train, MIT_split_val = random_split(MIT_split_train, [train_size, val_size])

    
    
    #for batch_idx, (images, labels, captions) in enumerate(test_dataloader):
    #    print(batch_idx, images.shape, labels, captions)

    KNN_METRICS = ["euclidean", "manhattan", "chebyshev", "minkowski"]


    def objective(trial):
    
        
        knn_metric = trial.suggest_categorical("knn_metric", KNN_METRICS)
        n_neighbors = trial.suggest_int("n_neighbors", 5, 50)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
        embed_size = trial.suggest_categorical("embed_size", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 2096])

        train_dataloader = DataLoader(MIT_split_train, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(MIT_split_val, batch_size=batch_size, shuffle=True)

        print("Looping through test dataloader...")
        test_dataloader = DataLoader(MIT_split_test, batch_size=batch_size, shuffle=True)

        print("\n\n\nInitializing the Model...")
        # CLASSIFICATION WITH RESNET, SO CROSS ENTROPY
        loss=torch.nn.CrossEntropyLoss()
        model = ResNet_embedding(embed_size=EMBED_SIZE, batch_size=batch_size, loss=loss, base_lr=BASE_LR)
        start_time = time()
        clf, train_labels = model.train_knn(train_dataloader, knn_metric, n_neighbors=n_neighbors)
        knn_time = time()-start_time
        avg_precision, mapk1, mapk5 = model.test(train_labels, test_dataloader, clf)
        print(f"\n\nObtained average precision: {avg_precision}")
        print(f"\nObtained mapk1: {mapk1} and mapk5: {mapk5}")


        ## SAVE MODEL:
        #SAVE_WEIGHTS_PATH = config["SAVE_WEIGHTS_PATH"]
        unique_filename = str(uuid.uuid4())
        #SAVE_WEIGHTS_PATH = os.path.join(SAVE_WEIGHTS_PATH, "task_a", unique_filename+".pt")
        #torch.save(model.state_dict(), SAVE_WEIGHTS_PATH)
        #print(f"Model Saved Successfully as {unique_filename}")

        # SAVE RESULTS
        # ------------------------------------------------------------------------
        SAVE_RESULTS_PATH = config["SAVE_RESULTS_PATH"]
        SAVE_RESULTS_PATH = os.path.join(SAVE_RESULTS_PATH, "task_a.json")
        result = {
            
            "weights_name": unique_filename,
            "average_precision": avg_precision,
            "mapk1": mapk1,
            "mapk5": mapk5,
            "sum_mapk1_mapk5": mapk1+mapk5,
            "embed_size": embed_size,
            "KNN_metric": knn_metric,
            "KNN_n_neighbors": n_neighbors,
            "KNN_fit_time": knn_time,
            "BATCH_SIZE": batch_size,
            "N_EPOCHS": NUM_OF_EPOCHS,
            "EMBED_SIZE": EMBED_SIZE,
            "PREPROCESS": save_transforms,
            "BASE_LR": BASE_LR

        }

        data = json.load(open(SAVE_RESULTS_PATH))
        # convert data to list if not
        if type(data) is dict:
            data = [data]

        # append new item to data lit
        data.append(result)

        # write list to file
        with open(SAVE_RESULTS_PATH, 'w') as outfile:
            json.dump(data, outfile, indent = 4)
        # ------------------------------------------------------------------------

        return mapk1+mapk5
    

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=300)






    


