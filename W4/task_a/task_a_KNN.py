import os
import sys

import io

import numpy as np
import math
import pickle
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE

from task_a import EmbeddingLayer, Model
from dataloader import Dataset

from matplotlib.lines import Line2D

distinct_colors = [
            [255, 0, 0],    # Red
            [0, 0, 255]   # Blue
]

def tsne_embeddings(model, dataloader, title="TSNE_plot_task_a"):

    '''
    model.embedding_model.eval()
    model.model_img.eval()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    #pbar = tqdm(enumerate(dataloader), total=math.ceil(len(dataloader) / 1000))

    all_imgs_embds = []
    all_captions_embds = []
    all_ids = []

    counter = 0

    # We will use the IDs for the TSNE plot
    for batch_idx, (anchor_img, captions, ids) in pbar:
        if counter > 300: break

        # Image Embeddings
        anchor_img = anchor_img.float() # (bs, 3, 224, 224)
        anchor_out = model.model_img(anchor_img) # (bs, 4096)

        # Text Embeddings
        text_vecs = []
        for caption in captions:
            word_vecs = []
            for word in caption.split():
                if word.lower() in model.model_txt:
                    word_vecs.append(torch.tensor(model.model_txt[word.lower()]))
            text_vecs.append(torch.stack(word_vecs).mean(dim=0))
        text_vecs = torch.stack(text_vecs).to(model.device)

        pos_embds = model.embedding_model.preforward_text(text_vecs) # (bs, 4096)


        # Add the Batch's imgs, captions and ID's to the full lists
        all_imgs_embds.extend(anchor_out)
        all_captions_embds.extend(pos_embds)
        all_ids.extend(ids)

        counter += 1
    '''


    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else: return super().find_class(module, name)

    ...
    #contents = pickle.load(f) becomes...
    #contents = CPU_Unpickler(f).load()
        
    with open('all_imgs_embds.pkl', 'rb') as f:
        #all_imgs_embds = torch.load(f, map_location=torch.device('cpu'))
        all_imgs_embds = CPU_Unpickler(f).load()
    with open('all_captions_embds.pkl', 'rb') as f:
        #all_captions_embds = torch.load(f, map_location=torch.device('cpu'))
        all_captions_embds = CPU_Unpickler(f).load()
    with open('all_ids.pkl', 'rb') as f:
        #all_ids = torch.load(f, map_location=torch.device('cpu'))
        all_ids = CPU_Unpickler(f).load()

    n_imgs, n_txt = len(all_imgs_embds), len(all_captions_embds)

    all_ids = [x.tolist() for x in all_ids]

    all_embds = all_imgs_embds + all_captions_embds
    all_embds = [x.detach().cpu().numpy() for x in all_embds]
    all_colors = [0]*n_imgs + [1]*n_txt
    all_colors_column = [[x] for x in all_colors]
    features_data = np.hstack([all_embds, all_colors_column])

    N_COMPONENTS = 2
    out_tsne = TSNE(n_components=N_COMPONENTS, verbose=1, metric='euclidean').fit_transform(features_data)

    df = pd.DataFrame(dict(x=out_tsne[:, 0], y=out_tsne[:, 1], label=all_colors))
    df['label'] = df['label'].replace({0: 'Images', 1: 'Text'})
    sns.set_style("whitegrid")
    sns.scatterplot(x="x", y="y", hue="label", data=df, legend=True)

    for ii, label in enumerate(all_ids + all_ids):
        plt.annotate(str(label), (out_tsne[ii, 0], out_tsne[ii, 1]), color="black", alpha=0.5, fontsize=4)
        
    plt.title(title)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    # plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Images'), 
                        # plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=5, label='Captions')], title='Legend')

    plt.savefig(f'./{title}.png', dpi=300, bbox_inches='tight')



def create_and_compute_KNN(fit_features_x, fit_features_y, features_to_predict, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(fit_features_x, fit_features_y)
    pred_labels = knn.predict(features_to_predict)
    pred_labels = np.array(pred_labels)

    return pred_labels







if __name__ == "__main__":

    '''
    # LOAD SAVED MODEL
    #========================================================
    WEIGHT_SAVE_PATH = './weights/model_img_task_a.pth'

    model = Model()
    #model.load_state_dict(torch.load(WEIGHT_SAVE_PATH))
    #========================================================
    '''
    
    dataset_val = Dataset(False)
    dataloader_val = DataLoader(dataset_val, drop_last=True, shuffle=True)
    

    tsne_embeddings(None, dataloader_val, title="prueba")


