import os
import sys
sys.path.append('./..') 
from torch.utils.data import DataLoader

import numpy as np
import json
from tqdm import tqdm
from PIL import Image

import torch
import torch.optim as optim
from torchvision import transforms

import fasttext
from dataloader import Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from tqdm import tqdm


# EMBEDDING LAYER
#=========================================================================================================
class EmbeddingLayer(torch.nn.Module):
    def __init__(self, embed_size):
        super(EmbeddingLayer, self).__init__()
        self.text_linear = torch.nn.Linear(300, embed_size)
        self.image_linear = torch.nn.Linear(4096, embed_size)
        self.activation = torch.nn.ReLU()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        #print(f'Lo que le llega en nuestro embedding {x["pool"].shape} {x["0"].shape} {x["1"].shape} {x["2"].shape}')
        x = x["pool"].flatten(start_dim=1)
        x = self.activation(x)
        x = self.image_linear(x)
        return x

    def preforward_text(self, x):
        return self.text_linear(x)
    
#=========================================================================================================


class Model():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading EmbeddingLayer...")
        self.embedding_model = EmbeddingLayer(embed_size=4096).to(self.device)

         # MODEL FOR IMGS
        #==========================================================
        print("Loading FasterRCNN...")
        self.model_img = fasterrcnn_resnet50_fpn(weights='COCO_V1').backbone
        self.model_img = torch.nn.Sequential(*list(self.model_img.children())[:], self.embedding_model)
        self.model_img.to(self.device)
        #==========================================================

        # MODEL FOR TEXT
        #==========================================================
        print("Loading FastText...")
        self.model_txt = fasttext.load_model('./../../mcv/C5/fasttext_wiki.en.bin')
        #==========================================================

        # LOSSES AND OPTIMIZERS
        #==========================================================
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=0.5, p=2, eps=1e-7)
        self.optimizer = optim.Adam(self.embedding_model.parameters(), lr=2e-5)
        #==========================================================

    def train_img_to_text(self, dataloader, n_epochs=3, save_path='./weigths/model_img_task_b_3epoch.pth') -> None:
        self.embedding_model.train()
        self.model_img.eval()

        for epoch in range(n_epochs):
            running_loss = []
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            # IDs not needed now, we will use them for the TSNE plots
            for batch_idx, (anchor_img, captions, _) in pbar:
                self.optimizer.zero_grad()

                # Image Embeddings
                anchor_img = anchor_img.float() # (bs, 3, 224, 224)
                anchor_out = self.model_img(anchor_img) # (bs, 4096)

                # Text Embeddings
                text_vecs = []
                for caption in captions:
                    word_vecs = []
                    for word in caption.split():
                        if word.lower() in self.model_txt:
                            word_vecs.append(torch.tensor(self.model_txt[word.lower()]))
                    text_vecs.append(torch.stack(word_vecs).mean(dim=0))
                text_vecs = torch.stack(text_vecs).to(self.device)

                text_embds = self.embedding_model.preforward_text(text_vecs) # (bs, 4096)
                neg_embds = torch.roll(anchor_out, shifts=-1, dims=0) # (bs, 4096)

                # Compute Triplet Loss
                loss = self.triplet_loss(text_embds, anchor_out, neg_embds)
                loss.backward()
                self.optimizer.step()
                running_loss.append(loss.item())

                pbar.set_description(f"Avg Running Loss: {np.mean(running_loss):.4f}")

            print(f'EPOCH {epoch} Avg Triplet Loss: {torch.Tensor(running_loss).mean()}')
                
        os.makedirs("./weights", exist_ok=True)
        torch.save(self.embedding_model.state_dict(), save_path)


if __name__ == '__main__':
    f = open('/ghome/group07/C5-W4/configs/task_a_train_config.json')
    config = json.load(f)
    train_model = config["train"]
    model = Model()
    print("Loading data...")
    dataset_train = Dataset(train_model)
    dataloader_train = DataLoader(dataset_train, batch_size=32, drop_last=True, shuffle=True)
    model.train_img_to_text(dataloader_train)






