
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
from dataloader import Dataset
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
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        #==========================================================

        # LOSSES AND OPTIMIZERS
        #==========================================================
        params = self.text_model.parameters()
        self.optimizer = optim.Adam(params, lr=1e-6, weight_decay=0.)

        N_AUGMENTATIONS: int = 3
        train_data_per_category_augmented: list[int] = [10*N_AUGMENTATIONS, 164*N_AUGMENTATIONS, 1264, 2932, 1353, 232*N_AUGMENTATIONS, 51*N_AUGMENTATIONS]
        n_elements_augmented: int = sum(train_data_per_category_augmented)

        class_weigths_augmented = [num/n_elements_augmented for num in train_data_per_category_augmented]
        class_weights_augmented = torch.tensor(class_weigths_augmented).to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights_augmented).to(self.device))
        #==========================================================



    def extract_features(self, dataloader, n_epochs=1) -> None:
        # Embedding Models
        # self.bert.eval()
        self.text_model.train()
        print("Entrado features ")
        print(f'Dataloader {dataloader}')

        for epoch in range(n_epochs):
            running_loss = []
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            for batch_idx, data in pbar:
                text, labels = data

                # text = text.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                
                # BERT
                #========================================================
                encoding = self.tokenizer.batch_encode_plus(
                    text,                    # List of input texts
                    padding=True,              # Pad to the maximum sequence length
                    truncation=True,           # Truncate to the maximum sequence length if necessary
                    return_tensors='pt',      # Return PyTorch tensors
                    add_special_tokens=True    # Add special tokens CLS and SEP
                )
                
                input_ids = encoding['input_ids']  # Token IDs
                attention_mask = encoding['attention_mask']  # Attention mask

                with torch.no_grad():
                    outputs = self.bert(input_ids, attention_mask=attention_mask)
                    word_embeddings = outputs.last_hidden_state  # This contains the embeddings
                
                # Compute the average of word embeddings to get the sentence embedding
                sentence_embedding = word_embeddings.mean(dim=1).to(self.device)  # Average pooling along the sequence length dimension
                #========================================================

                outputs = self.text_model.forward(sentence_embedding)

                # FEATURES: OUTPUTS
                pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

                print(f'Captions: {text}')
                print(f'Embeddings: {sentence_embedding.shape}')
                print(f'Features: {outputs.shape}')
                print()

                # Compute Triplet Loss
                # loss = self.loss_fn()
                # loss.backward()
                self.optimizer.step()
                # running_loss.append(loss.item())


            # print(f'EPOCH {epoch} Avg Triplet Loss: {torch.Tensor(running_loss).mean()}')
                

        # SAVE EMBEDDINGS
        save_path_img = './weights/image_model_task_b_1epoch.pth'
        save_path_txt = './weights/text_model_task_b_1epoch.pth'
        os.makedirs("./weights", exist_ok=True)
        torch.save(self.model_img.state_dict(), save_path_img)
        torch.save(self.text_model.state_dict(), save_path_txt)


if __name__ == '__main__':
    batch_size = 256

    model = Model()
    print("Loading data...")
    train_dir, valid_dir, test_dir = get_data_sets_path('/ghome/group07/C5-W6/First_Impressions_v3_multimodal')
    # DATALOADER =========================================================
    train_dataset = Dataset(regime='train')
    val_dataset = Dataset(regime='val')
    test_dataset = Dataset(regime='test')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=2, shuffle=True)

    # ====================================================================
    model.extract_features(train_dataloader)

