
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

from torchmetrics import Accuracy

from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f'The device we will be working on is: {device}')



CLASSES = ['Opencountry','coast','forest','highway','inside_city','mountain','street','tallbuilding']

IMG_CHANNELS = 3
OUTPUT_SHAPE = 8
#N_EPOCHS = 75

class net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.training_step_outputs = []   # save outputs in each batch to compute metric overall epoch
        self.training_step_targets = []   # save targets in each batch to compute metric overall epoch
        self.training_step_loss = []
        self.val_step_outputs = []        # save outputs in each batch to compute metric overall epoch
        self.val_step_targets = []        # save targets in each batch to compute metric overall epoch
        self.val_step_loss = []

        self.best_validation_acc = 0

        # log hyperparameters
        self.save_hyperparameters()

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

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=OUTPUT_SHAPE)

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



    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        # GET AND SAVE OUTPUTS AND TARGETS PER BATCH
        #y_pred = y_hat.argmax(dim=1)
        #y_true = y.cpu()
        
        self.training_step_outputs.extend(preds.cpu().numpy())
        self.training_step_targets.extend(y.cpu().numpy())
        self.training_step_loss.append(loss.cpu().item())

        return loss

    def on_train_epoch_end(self):
        train_all_loss = self.training_step_loss
        train_all_outputs = self.training_step_outputs
        train_all_targets = self.training_step_targets

        print(train_all_loss)

        acc = self.accuracy(torch.tensor(train_all_outputs),torch.tensor(train_all_targets))
        m_loss = np.mean(train_all_loss)

        self.log('train_loss', m_loss, on_step=False, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, logger=True)

        wandb.log({"epoch": self.current_epoch + 1, "train_loss": m_loss, "train_acc": acc})

        self.training_step_loss = []
        self.training_step_outputs = []
        self.training_step_targets = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        
        self.val_step_outputs.extend(preds.cpu().numpy())
        self.val_step_targets.extend(y.cpu().numpy())
        self.val_step_loss.append(loss.cpu().detach().numpy())

        return loss
    
    def on_validation_epoch_end(self):
        val_all_loss = self.val_step_loss
        val_all_outputs = self.val_step_outputs
        val_all_targets = self.val_step_targets

        acc = self.accuracy(torch.tensor(val_all_outputs),torch.tensor(val_all_targets))
        m_loss = np.mean(val_all_loss)

        if acc >= self.best_validation_acc:
            self.best_validation_acc = acc
            wandb.log({"Best Validation": acc})

        self.log('val_loss', m_loss, on_step=False, on_epoch=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, logger=True)

        wandb.log({"epoch": self.current_epoch + 1, "val_loss": m_loss, "val_acc": acc})

        self.val_step_loss = []
        self.val_step_outputs = []
        self.val_step_targets = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

MODEL_PATH = "/ghome/group07/C5/3path_lightning/modelos_C5/batch_64_aug3.ckpt"

loaded_model = net.load_from_checkpoint(MODEL_PATH)


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

trainer = Trainer(resume_from_checkpoint=MODEL_PATH)
print(trainer.test(loaded_model, dataloaders=test_loader))
trainer.test(loaded_model, dataloaders=test_loader)

calculate_accuracy_and_loss(loaded_model, test_loader, device)
