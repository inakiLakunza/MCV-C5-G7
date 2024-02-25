
#import tensorflow as tf
#from tensorflow.keras.models import load_model
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import RocCurveDisplay
import numpy as np
from utils import *

import os
os.environ["KERAS_BACKEND"] = "torch"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import keras
from keras.utils import plot_model

import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f'The device we will be working on is: {device}')



CLASSES = ['Opencountry','coast','forest','highway','inside_city','mountain','street','tallbuilding']

IMG_CHANNELS = 3
OUTPUT_SHAPE = 8
#N_EPOCHS = 75

class net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_block_first = nn.Sequential(
            nn.Conv2d(in_channels=IMG_CHANNELS,
                      out_channels=8,
                      kernel_size=9,  
                      stride=3,
                      padding=0),  
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=8,
                      out_channels=8,
                      kernel_size=5,  
                      stride=1,
                      padding=0),  
            nn.BatchNorm2d(8),
            nn.ReLU(),
        
            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=3, 
                      stride=1,
                      padding=0), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        
        self.conv_block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=8,
                      out_channels=8,
                      kernel_size=5, 
                      stride=1,
                      padding=0),  
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=3, 
                      stride=1,
                      padding=0),  
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.conv_block3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=8,
                      out_channels=8,
                      kernel_size=5,  
                      stride=1,
                      padding=0),  
            nn.BatchNorm2d(8),
            nn.ReLU(),
        
            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=3, 
                      stride=1,
                      padding=0), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        
        self.dense_block = nn.Sequential(
            nn.Flatten()
        )
        
        self.output = nn.Sequential(
            nn.Linear(in_features=48*1*1, 
                      out_features=OUTPUT_SHAPE)
        )

    def forward(self, x):
        # INITIAL CONVOLUTION
        x = self.conv_block_first(x)
        x = self.pool2(x)

        # PATH 1
        x1 = self.conv_block1(x)
        x1 = nn.functional.avg_pool2d(x1, x1.size()[2:])
        
        # PATH 2
        x2 = self.conv_block2(x)
        x2 = nn.functional.avg_pool2d(x2, x2.size()[2:])
        
        # PATH 3
        x3 = self.conv_block3(x)
        x3 = nn.functional.avg_pool2d(x3, x3.size()[2:])

        # COMBINATION OF PATHS
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        out = torch.cat((x1, x2, x3), 1)
        #print(f'Output shape of out: {out.shape}')
        out = out.view(-1, 48*1*1)

        clas = self.output(out)
        return clas


MODEL_PATH = "/ghome/group07/C5/3path_model/modelos_C5/torch_set3_b16.pt"

loaded_model = net().to(device)
loaded_model.load_state_dict(torch.load(MODEL_PATH))


all_y = []
all_preds = []


DATASET_DIR = '/ghome/group07/C5/MIT_small_augments/small_bigger_3/'
test_path = DATASET_DIR+'/test'
batch_size = 16

transform = transforms.Compose([
    #transforms.Resize((256, 256)),  # Resize images to a fixed size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
])

test_data = datasets.ImageFolder(root=test_path, transform=transform)

test_loader = torch.utils.data.DataLoader(
                dataset=test_data,
                batch_size=batch_size,
                shuffle=False)

for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)


calculate_accuracy_and_loss(loaded_model, test_loader, device)



'''
preds = []
pred_scores = []
targets = []

for cls in os.listdir(test_path):
    subdirectory_path = os.path.join(VAL_DIR, cls)
    image_paths = [os.path.join(subdirectory_path, image_file) for image_file in os.listdir(subdirectory_path)]
    for image_path in image_paths:
        img = load_and_preprocess_image(image_path=image_path,
                                        target_size=(img_size, img_size))
 
        pred = list(loaded_model.predict(img)[0])
        pred_score = max(pred)

        pred = pred.index(max(pred))
        pred = CLASSES[pred]

        preds.append(pred)
        pred_scores.append(pred_score)
        targets.append(cls)


# plot preds



create_confusion_matrix(targets, preds)
metrics = get_metrics(targets, preds)
print(metrics)
#create and save auroc curve
RocCurveDisplay.from_predictions( targets, pred_scores)
plt.savefig('auroc.png')
'''