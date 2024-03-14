import torch
import sys
import numpy as np
import json
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, random_split
from dataset import Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import average_precision_score
from model_class import Faster_embedding
import pickle as pkl
#from losses import ContrastiveLoss
import utils
import random
import matplotlib.pyplot as plt
from pycocotools import mask
import cv2
import tqdm
import pickle
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import tqdm


# MODEL ====================================================================

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, embed_size):
        super(EmbeddingLayer, self).__init__()
        self.linear = torch.nn.Linear(4096, embed_size)
        self.activation = torch.nn.ReLU()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        #print(f'Lo que le llega en nuestro embedding {x["pool"].shape} {x["0"].shape} {x["1"].shape} {x["2"].shape}')

        x = x["pool"].flatten()
        x = self.activation(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    f = open('./configs/task_e_train_config.json')
    config = json.load(f)
    generate_data_dicts = False
    generate_dataloader = False

    PATH_PARENT_DIRECTORY = "/ghome/group07/mcv/datasets/C5/COCO"
    PATH_TRAINING_SET = os.path.join(PATH_PARENT_DIRECTORY, "train2014")
    PATH_VAL_SET = os.path.join(PATH_PARENT_DIRECTORY, "val2014")
    PATH_INSTANCES_TRAIN = os.path.join(PATH_PARENT_DIRECTORY, "instances_train2014.json")
    PATH_INSTANCES_VAL = os.path.join(PATH_PARENT_DIRECTORY, "instances_val2014.json")
    RETRIEVAL_ANNOTATIONS = "/ghome/group07/mcv/datasets/C5/COCO/mcv_image_retrieval_annotations.json"
    FT_DATASET_NAME = "coco"

    path_ids_train = './data/custom_ids/train.pkl'
    path_labels_train = './data/custom_labels/train.pkl'
    path_ids_test = './data/custom_ids/test.pkl'
    path_labels_test = './data/custom_labels/test.pkl'

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    totensor = transforms.Compose([transforms.Resize((224, 224)), transforms.PILToTensor()])
    triplet_loss = torch.nn.TripletMarginLoss(margin=0.5, p=2, eps=1e-7)
    model = fasterrcnn_resnet50_fpn(weights='COCO_V1').backbone
    embed = EmbeddingLayer(embed_size=2048)
    model = torch.nn.Sequential(*list(model.children())[:], embed)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    # Antes
    #model = models.resnet18(weights=True)
    # model = torch.nn.Sequential(*list(model.children())[:-1], embed)

    f = open(RETRIEVAL_ANNOTATIONS)
    object_annotations = json.load(f)

    # Generate data dicts if true...
    if generate_data_dicts:
        os.makedirs('./data/custom_ids', exist_ok=True)
        os.makedirs('./data/custom_labels', exist_ok=True)
        utils.save_custom_data(object_annotations["train"], path_ids_train, path_labels_train)
        utils.save_custom_data(object_annotations["test"], path_ids_test, path_labels_test)

    with open(path_ids_train, "rb") as f:
        ids_train = pkl.load(f)

    with open(path_labels_train, "rb") as f:
        labels_train = pkl.load(f)
    
    
    # Generate train dict (anchor,  positive,  negative,  anchor labels)
    if generate_dataloader:
        os.makedirs('./data/dataloader', exist_ok=True)
        dataloader_train = utils.create_image_loader(labels_train)
        with open('./data/dataloader/train_1_1.pkl', "wb") as f:
            pkl.dump(dataloader_train, f)
    
    with open('./data/dataloader/train_1_1.pkl', "rb") as f:
        dataloader_train = pkl.load(f)


    # TODO: continue here with the training loop from iñaki notebook
    model.train()
    num_epochs = 3
    for epoch in range(num_epochs):
        running_loss = []
        for anchor, positive, negative, anchor_labels in tqdm.tqdm(dataloader_train):
            # print(f'anchor: {anchor}')
            # print(f'positive: {positive}')
            # print(f'negative: {negative}')
            # print(f'anchor_labels: {anchor_labels}')

            optimizer.zero_grad()

            # Read Images
            anchor_path = os.path.join(PATH_TRAINING_SET, 'COCO_train2014_' + str(anchor).zfill(12) + '.jpg')
            pos_path = os.path.join(PATH_TRAINING_SET, 'COCO_train2014_' + str(positive).zfill(12) + '.jpg')
            neg_path = os.path.join(PATH_TRAINING_SET, 'COCO_train2014_' + str(negative).zfill(12) + '.jpg')
            anchor_img = totensor(Image.open(anchor_path).convert('RGB')).to(device)
            pos_img = totensor(Image.open(pos_path).convert('RGB')).to(device)
            neg_img = totensor(Image.open(neg_path).convert('RGB')).to(device)

            # Embeddings
            anchor_img = anchor_img.float().unsqueeze(0)
            pos_img = pos_img.float().unsqueeze(0)
            neg_img = neg_img.float().unsqueeze(0) 
            anchor_out = model(anchor_img)
            pos_out = model(pos_img)
            neg_out = model(neg_img)

            # Compute Triplet Loss
            loss = triplet_loss(anchor_out, pos_out, neg_out)
            # print(loss)
            loss.backward()
            optimizer.step()
            running_loss.append(loss)
        
        print(f'EPOCH {epoch} Avg Triplet Loss: {sum(running_loss) / len(running_loss)}')
            
    
    SAVE_PATH = "/ghome/group07/C5-W3/task_e/saved_models/model_emb2048_margin05_p2_epochs2.pth"
    torch.save(model.state_dict(), SAVE_PATH)









    sys.exit()
          



    # path_ids_train = './data/custom_ids/train.pkl'
    # path_labels_train = './data/custom_ids/train.pkl'
    # utils.save_custom_data(object_annotations["train"], path_ids_train, path_labels_train)


    '''
    test_ids=[]
    for i in enumerate(object_labelations['test']):
        for j in range(len(object_labelations['test'][i[1]])):
            test_ids.append(object_labelations['test'][i[1]][j])
    test_ids=list(set(test_ids))


    print("len de test: "+str(len(test_ids)))

    val_ids=[]
    for i in enumerate(object_labelations['val']):
        for j in range(len(object_labelations['val'][i[1]])):
            val_ids.append(object_labelations['val'][i[1]][j])
    val_ids=list(set(val_ids))

    print("len de val:" +str(len(val_ids)))
    '''
    train_ids=[]
    train_dict = {}
    for object_label in object_labelations['train']:
        object_labels.append(int(object_label))
        for j in range(len(object_labelations['train'][object_label])):
            train_ids.append(object_labelations['train'][object_label][j])
            img_id = object_labelations['train'][object_label][j]
            #img = Image.open(os.path.join(PATH_TRAINING_SET, 'COCO_train2014_' + str(img_id).zfill(12) + '.jpg')).convert('RGB')
            img = object_label
            if img_id not in train_dict.keys():
                train_dict[img_id] = [img]
            else:
                train_dict[img_id].append(img)
        print(train_dict)
        sys.exit()
        train_imgs.append((img, label))
    #print(train_imgs)
    #train_ids=list(set(train_ids))
    object_labels = sorted(object_labels)
    print("len labels", len(object_labels))
    for label in object_labels:
        print(label)

    
    sys.exit()

    print("len de train: "+str(len(train_ids)))
    database_ids=[]
    for i in enumerate(object_labelations['database']):
        for j in range(len(object_labelations['database'][i[1]])):
            database_ids.append(object_labelations['database'][i[1]][j])
    database_ids=list(set(database_ids))

    print("len de db: "+str(len(database_ids)))


    
    #register_coco_instances("my_dataset_train", {}, PATH_INSTANCES_TRAIN, PATH_TRAINING_SET)
    #register_coco_instances("my_dataset_val", {}, PATH_INSTANCES_VAL, PATH_VAL_SET)
    #register_coco_instances("my_dataset_database", {}, "json_object_labelation_val.json", "path/to/image/dir")
    #register_coco_instances("my_dataset_test", {}, "json_object_labelation_val.json", "path/to/image/dir")

  
    my_train = MetadataCatalog.get("my_dataset_val")
    dataset_dicts = DatasetCatalog.get("my_dataset_val")

    for d in random.sample(dataset_dicts, 3):
        img = cv.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata= my_train)
        vis = visualizer.draw_dataset_dict(d)
        cv.imwrite('/ghome/group07/C5-W3/images_a/image_COCO.jpg',vis.get_image()[:, :, ::-1])
    
 
