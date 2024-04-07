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
class ImageEmbeddingLayer(torch.nn.Module):
    def __init__(self):
        super(ImageEmbeddingLayer, self).__init__()
        self.image_linear = torch.nn.Linear(4096, 1000)
        self.activation = torch.nn.ReLU()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        x = x["pool"].flatten(start_dim=1)
        x = self.activation(x)
        x = self.image_linear(x)
        return x
    
class TextEmbeddingLayer(torch.nn.Module):
    def __init__(self):
        super(TextEmbeddingLayer, self).__init__()
        self.text_linear = torch.nn.Linear(300, 1000)
        self.activation = torch.nn.ReLU()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        x = self.activation(x)
        x = self.text_linear(x)
        return x
    
#=========================================================================================================


class Model():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading EmbeddingLayers...")
        self.image_model = ImageEmbeddingLayer().to(self.device)
        self.text_model = TextEmbeddingLayer().to(self.device)

         # MODEL FOR IMGS
        #==========================================================
        print("Loading FasterRCNN...")
        self.model_img = fasterrcnn_resnet50_fpn(weights='COCO_V1').backbone
        self.model_img = torch.nn.Sequential(*list(self.model_img.children())[:], self.image_model)
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
        params = list(self.image_model.parameters()) + list(self.text_model.parameters())
        self.optimizer = optim.AdamW(params, lr=2e-5, weight_decay=0.)
        #==========================================================

    def train_img_to_text(self, dataloader, n_epochs=4, save_path_img='./weights/image_model_task_a_4epoch_embed_1000.pth', save_path_txt='./weights/text_model_task_a_1epoch_embed_1000.pth') -> None:
        # Embedding Models
        self.image_model.train()
        self.text_model.train()

        # Faster
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

                pos_embds = self.text_model.forward(text_vecs) # (bs, 1000)
                neg_embds = torch.roll(pos_embds, shifts=-1, dims=0) # (bs, 1000)

                # Compute Triplet Loss
                loss = self.triplet_loss(anchor_out, pos_embds, neg_embds)
                loss.backward()
                self.optimizer.step()
                running_loss.append(loss.item())

                pbar.set_description(f"Avg Running Loss: {np.mean(running_loss):.4f}")

            print(f'EPOCH {epoch} Avg Triplet Loss: {torch.Tensor(running_loss).mean()}')
                
        os.makedirs("./weights", exist_ok=True)
        torch.save(self.model_img.state_dict(), save_path_img)
        torch.save(self.text_model.state_dict(), save_path_txt)

        # SAVE EMBEDDINGS


if __name__ == '__main__':
    f = open('/ghome/group07/C5-W4/configs/task_a_train_config.json')
    config = json.load(f)
    train_model = config["train"]
    model = Model()
    print("Loading data...")
    dataset_train = Dataset(train_model)
    dataloader_train = DataLoader(dataset_train, batch_size=32, drop_last=True, shuffle=True)
    model.train_img_to_text(dataloader_train)






