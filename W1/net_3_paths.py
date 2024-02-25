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

from torchviz import make_dot

import wandb

wandb.init(
    project="different-paths",
    name="torch_set3_b64",
    config={
    "architecture": "CNN",
    "dataset": "MIT_small_augments/small_bigger_3",
    "epochs": 75,
    "batch_size": 64,
    }
)

# log the minimum val_loss and max val_acc
wandb.define_metric("val_loss", summary="min")
wandb.define_metric("val_acc", summary="max")

print(torch.cuda.device_count())

device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f'The device we will be working on is: {device}')


IMG_CHANNELS = 3
OUTPUT_SHAPE = 8
N_EPOCHS = 75


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

model = net().to(device)
summary(model, (3, 256, 256))
wandb.log({"Model": summary(model, (3, 256, 256))})


optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()


DATASET_DIR = '/ghome/group07/C5/MIT_small_augments/small_bigger_3/'
batch_size = 64


transform = transforms.Compose([
    #transforms.Resize((256, 256)),  # Resize images to a fixed size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
])

train_path = DATASET_DIR+'/train'
test_path = DATASET_DIR+'/test'

train_data = datasets.ImageFolder(root=train_path, transform=transform)
test_data = datasets.ImageFolder(root=test_path, transform=transform)

train_loader = torch.utils.data.DataLoader(
                    dataset=train_data,
                    batch_size=batch_size,
                    shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_data,
                batch_size=batch_size,
                shuffle=False)

for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)

for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)


def accuracy_fn(y_true, y_pred):
    correct = troch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc

def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    total_samples = 0
    correct_predictions = 0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        running_loss += loss

        loss.backward()

        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Adjust learning weights
        optimizer.step()

    train_loss = running_loss/ len(train_loader)
    train_acc = correct_predictions / total_samples

    return train_loss, train_acc

# Initializing in a separate cell so we can easily add more epochs to the same run
epoch_number = 0

best_tloss = 1_000_000.

def calculate_accuracy_and_loss(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode

    correct_predictions = 0
    total_samples = 0

    loss = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get the predicted class labels
            _, predicted = torch.max(outputs, 1)

            # Update counts
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            loss += loss_fn(outputs, labels)

    accuracy = correct_predictions / total_samples
    loss /= len(data_loader)
    
    return accuracy, loss

for epoch in range(N_EPOCHS):
    start_time = time.time()
    model.train(True)
    train_loss, train_acc = train_one_epoch(epoch_number)

    model.eval()
    val_acc, val_loss = calculate_accuracy_and_loss(model, test_loader, device=device)

    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f'[Epoch: {epoch + 1}/{N_EPOCHS} ({epoch_duration:.2f}s)] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

    wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc, "val_loss": val_loss})

    # Track best performance, and save the model's state
    if val_loss < best_tloss:
        best_tloss = val_loss
        model_path = './modelos_C5/torch_set3_b64.pt'
        wandb.log({"Best Validation": val_acc})
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
