
import os
import sys
sys.path.append('./..') 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import pickle
from torchvision.transforms import v2
from text_dataloader import Dataset
import tqdm

import torch
import torch.optim as optim
from torchvision import transforms

from transformers import BertTokenizer, BertModel


from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import ConcatDataset, DataLoader

import librosa
import librosa.display

import noisereduce as nr

from tqdm import tqdm



# EMBEDDING LAYER
#=========================================================================================================
class TextEmbeddingLayer(torch.nn.Module):
    def __init__(self):
        super(TextEmbeddingLayer, self).__init__()
        # BERT EMBEDDING SIZE: 768
        self.text_linear = torch.nn.Linear(768, 512)
        self.final_linear = torch.nn.Linear(512, 7)
        self.activation = torch.nn.ReLU()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        x = self.text_linear(x)
        x = self.activation(x)
        x = self.final_linear(x)
        return x
#=========================================================================================================


class Model():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading EmbeddingLayer...")
        self.text_model = TextEmbeddingLayer().to(self.device)

        # MODEL FOR TEXT
        #==========================================================
        print("Loading BERT...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        #==========================================================

        # LOSSES AND OPTIMIZERS
        #==========================================================
        params = self.text_model.parameters()
        self.text_optimizer = optim.Adam(params, lr=1e-6, weight_decay=0.)

        N_AUGMENTATIONS: int = 3
        train_data_per_category_augmented: list[int] = [10, 164, 1264, 2932, 1353, 232, 51]
        n_elements_augmented: int = sum(train_data_per_category_augmented)

        class_weigths_augmented = [num/n_elements_augmented for num in train_data_per_category_augmented]
        class_weights_augmented = torch.tensor(class_weigths_augmented).to(self.device)
        self.text_loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights_augmented).to(self.device))
        #==========================================================

        self.best_acc = 0



    def train(self, dataloader, val_dataloader, n_epochs=1) -> None:
        for epoch in range(n_epochs):
            self.text_model.train()
            running_loss = []
            running_acc = []
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            for batch_idx, data in pbar:
                texts, labels = data

                labels = labels.to(self.device)
                labels = torch.add(labels, -1)

                assert labels.min() >= 0 and labels.max() <= 6, "Label indices out of range"

                self.text_optimizer.zero_grad()
                
                # BERT
                #========================================================
                encoding = self.tokenizer.batch_encode_plus(
                    texts,                    # List of input texts
                    padding=True,              # Pad to the maximum sequence length
                    truncation=True,           # Truncate to the maximum sequence length if necessary
                    return_tensors='pt',      # Return PyTorch tensors
                    add_special_tokens=True    # Add special tokens CLS and SEP
                )
                
                input_ids = encoding['input_ids'].to(self.device)  # Token IDs
                attention_mask = encoding['attention_mask'].to(self.device)  # Attention mask

                with torch.no_grad():
                    outputs = self.bert(input_ids, attention_mask=attention_mask)
                    word_embeddings = outputs.last_hidden_state  # This contains the embeddings
                
                # Compute the average of word embeddings to get the sentence embedding
                sentence_embedding = word_embeddings.mean(dim=1).to(self.device)  # Average pooling along the sequence length dimension
                #========================================================

                outputs = self.text_model.forward(sentence_embedding)

                # Predict
                pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

                # Compute Loss
                text_loss = self.text_loss_fn(outputs, labels)
                text_loss.backward()
                self.text_optimizer.step()

                # Compute Accuracy
                correct_predictions = (pred_class == labels).float()
                accuracy = correct_predictions.sum() / len(labels)

                running_loss.append(text_loss)
                running_acc.append(accuracy)

            loss, acc = self.test(val_dataloader)
            print(f'EPOCH {epoch} |  Train Loss: {torch.Tensor(running_loss).mean().item()}  Train Acc: {torch.Tensor(running_acc).mean().item()}    Val Loss: {loss}  Val Acc: {acc}')
            
            if acc > self.best_acc:
                save_path_txt = './weights/text_model.pth'
                os.makedirs("./weights", exist_ok=True)
                torch.save(self.text_model.state_dict(), save_path_txt)
                self.best_acc = acc


    def test(self, dataloader):
        self.text_model.eval()
        loss = 0
        accuracy = 0
        batches = 0
        with torch.no_grad():
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            for batch_idx, data in pbar:
                texts, labels = data

                labels = labels.to(self.device)
                labels = torch.add(labels, -1)

                assert labels.min() >= 0 and labels.max() <= 6, "Label indices out of range"
                
                # BERT
                #========================================================
                encoding = self.tokenizer.batch_encode_plus(
                    texts,                    # List of input texts
                    padding=True,              # Pad to the maximum sequence length
                    truncation=True,           # Truncate to the maximum sequence length if necessary
                    return_tensors='pt',      # Return PyTorch tensors
                    add_special_tokens=True    # Add special tokens CLS and SEP
                )
                
                input_ids = encoding['input_ids'].to(self.device)  # Token IDs
                attention_mask = encoding['attention_mask'].to(self.device)  # Attention mask

                outputs = self.bert(input_ids, attention_mask=attention_mask)
                word_embeddings = outputs.last_hidden_state  # This contains the embeddings
                
                # Compute the average of word embeddings to get the sentence embedding
                sentence_embedding = word_embeddings.mean(dim=1).to(self.device)  # Average pooling along the sequence length dimension
                #========================================================

                outputs = self.text_model.forward(sentence_embedding)

                # Predict
                pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

                # Compute Loss
                loss += self.text_loss_fn(outputs, labels).item()

                # Compute Accuracy
                correct_predictions = (pred_class == labels).float()
                accuracy += correct_predictions.sum() / len(labels)
                batches += 1

        average_loss = loss / batches
        average_accuracy = accuracy / batches

        return average_loss, average_accuracy      



if __name__ == '__main__':
    batch_size = 256

    model = Model()
    print("Loading data...")
    # DATALOADER =========================================================
    train_dataset = Dataset(regime='train')
    val_dataset = Dataset(regime='val')
    test_dataset = Dataset(regime='test')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=2, shuffle=True)

    # ====================================================================
    model.train(train_dataloader, val_dataloader, n_epochs=300)
    print(f'\nFINAL TEST ==============================')
    loss, acc = model.test(test_dataloader)
    print(f'Loss: {loss} Acc: {acc}')

