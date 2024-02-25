import os,sys
import numpy as np

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

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

CLASSES = ['Opencountry', 'coast','forest','highway','inside_city','mountain','street','tallbuilding']


def get_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
 
    return accuracy, recall, precision, f1  

def create_confusion_matrix(targets, preds, save_path='confusion_matrix.png', classes=None):

    if classes == None:
        classes = list(set(targets))
    cf_matrix = confusion_matrix(targets, preds, normalize='true')
    df_cm = pd.DataFrame(cf_matrix, index = [CLASSES[i] for i in classes],
                    columns = [CLASSES[i] for i in classes])
    
    plt.figure(figsize = (12,7))
    plt.title("Confusion Matrix PyTorch")       
    sn.heatmap(df_cm, annot=True)
    plt.savefig(save_path)
    plt.close()    


def ROC_AUC(save_path, test_labels, all_pred_scores):
    classes = np.unique(CLASSES).tolist()
    test_labels_binarized = label_binarize(test_labels, classes=classes)

    fpr = dict()
    tpr = dict()
    avg = dict()
    roc_auc = dict()

    all_pred_scores = np.array(all_pred_scores)
    all_targets = np.array(test_labels)

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(CLASSES)):  # Assuming num_classes is defined
        fpr[i], tpr[i], _ = roc_curve((all_targets == i).astype(int), all_pred_scores[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    # Set the Seaborn style for a cleaner look
    sn.set(style="darkgrid", palette="muted")
    colors = sn.color_palette(n_colors=len(CLASSES))
    plt.figure(figsize=(8, 8))
    for i in range(len(CLASSES)):
        plt.plot(fpr[i], tpr[i], label=f'{CLASSES[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve PyTorch', fontsize=17)
    plt.legend(loc="lower right")
    plt.savefig(save_path)



def calculate_accuracy_and_loss(model, data_loader, device, save_conf='confusion_matrix.png', save_roc='auroc.png'):
    model.eval()  # Set the model to evaluation mode

    all_preds = []
    all_pred_scores = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get the predicted class labels
            #_, predicted = torch.max(outputs, 1)
            
            softmaxed = F.softmax(outputs, dim=1)
            #print(predicted)
            #print(outputs)
            #print(softmaxed)

            # Get prediction scores
            pred_scores, predicted = torch.max(softmaxed, 1)

            all_preds.extend(predicted.cpu().numpy())
            for inside_list in softmaxed:
                all_pred_scores.append(inside_list.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            #print(all_preds)
            #print(all_pred_scores)
            #print(all_targets)
    
    create_confusion_matrix(all_targets, all_preds, save_path=save_conf)
    metrics = get_metrics(all_targets, all_preds)
    print(metrics)
    #create and save auroc curve
    #RocCurveDisplay.from_predictions(all_targets, all_pred_scores, multi_class='ovr')
    ROC_AUC(save_roc, all_targets, all_pred_scores)
    #plt.savefig(save_roc)