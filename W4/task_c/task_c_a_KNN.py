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



def use_pickle_info(img_embds_pickle_path,
                    caption_embds_pickle_path,
                    ids_pickle_path):
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else: return super().find_class(module, name)

    #contents = pickle.load(f) becomes...
    #contents = CPU_Unpickler(f).load()
        
    with open(img_embds_pickle_path, 'rb') as f:
        #all_imgs_embds = torch.load(f, map_location=torch.device('cpu'))
        all_imgs_embds = CPU_Unpickler(f).load()
    with open(caption_embds_pickle_path, 'rb') as f:
        #all_captions_embds = torch.load(f, map_location=torch.device('cpu'))
        all_captions_embds = CPU_Unpickler(f).load()
    with open(ids_pickle_path, 'rb') as f:
        #all_ids = torch.load(f, map_location=torch.device('cpu'))
        all_ids = CPU_Unpickler(f).load()

    return all_imgs_embds, all_captions_embds, all_ids



def get_embeddings(img_model, txt_model, dataloader,
                   max_samples=100,
                   use_saved_pickles = False,
                   wanted_embeds = "task_b",
                   save_pickles=False,
                   pickle_suffix=""):
    
    if use_saved_pickles:
        
        PICKLE_ROOT = os.path.join('./', 'pickles', wanted_embeds)

        img_embds_pickle_path = os.path.join(PICKLE_ROOT, 'imgs.pkl')
        caption_embds_pickle_path = os.path.join(PICKLE_ROOT, 'captions.pkl')
        ids_pickle_path = os.path.join(PICKLE_ROOT, 'ids.pkl')

        all_imgs_embds, all_captions_embds, all_ids = use_pickle_info(img_embds_pickle_path, caption_embds_pickle_path, ids_pickle_path)

    else:

        img_model.eval()
        txt_model.eval()

        pbar = tqdm(enumerate(dataloader), total=max_samples)
        #pbar = tqdm(enumerate(dataloader), total=math.ceil(len(dataloader) / 1000))

        all_imgs_embds = []
        all_captions_embds = []
        all_ids = []

        counter = 0

        # We will use the IDs for the TSNE plot
        for batch_idx, (anchor_img, captions, ids) in pbar:
            if counter > max_samples: break

            # Image Embeddings
            anchor_img = anchor_img.float() # (bs, 3, 224, 224)
            anchor_out = img_model(anchor_img) # (bs, 4096)

            # BERT
            #========================================================
            encoding = model.tokenizer.batch_encode_plus(
                captions,                    # List of input texts
                padding=True,              # Pad to the maximum sequence length
                truncation=True,           # Truncate to the maximum sequence length if necessary
                return_tensors='pt',      # Return PyTorch tensors
                add_special_tokens=True    # Add special tokens CLS and SEP
            )
            
            input_ids = encoding['input_ids']  # Token IDs
            attention_mask = encoding['attention_mask']  # Attention mask

            with torch.no_grad():
                outputs = model.model_txt(input_ids, attention_mask=attention_mask)
                word_embeddings = outputs.last_hidden_state  # This contains the embeddings
            
            # Compute the average of word embeddings to get the sentence embedding
            sentence_embedding = word_embeddings.mean(dim=1).to(model.device)   # Average pooling along the sequence length dimension
            pos_embds = model.embedding_model.preforward_text(sentence_embedding) # (bs, 4096)
            #========================================================



            # Add the Batch's imgs, captions and ID's to the full lists
            all_imgs_embds.extend(anchor_out)
            all_captions_embds.extend(pos_embds)
            all_ids.extend(ids)

            counter += 1

        if save_pickles:
            os.makedirs(f"./pickles/{wanted_embeds}", exist_ok=True)

            with open(f"./pickles/{wanted_embeds}/captions{pickle_suffix}.pkl", "wb") as file:
                pickle.dump(all_captions_embds, file)
            with open(f"./pickles/{wanted_embeds}/ids{pickle_suffix}.pkl", "wb") as file:
                pickle.dump(all_ids, file)
            with open(f"./pickles/{wanted_embeds}/imgs{pickle_suffix}.pkl", "wb") as file:
                pickle.dump(all_imgs_embds, file)

    return all_imgs_embds, all_captions_embds, all_ids





def tsne_embeddings(img_model, txt_model, dataloader, title="TSNE_plot_task_c_a",
                    max_samples=100,
                    use_saved_pickles = False,
                    wanted_embeds = "task_c_a",
                    save_pickles=True,
                    pickle_suffix=""
                    ):

    all_imgs_embds, all_captions_embds, all_ids = get_embeddings(img_model, txt_model, dataloader,
                                                                 max_samples=max_samples,
                                                                 use_saved_pickles=use_saved_pickles,
                                                                 wanted_embeds=wanted_embeds,
                                                                 save_pickles=save_pickles,
                                                                 pickle_suffix=pickle_suffix
                                                                )

        

    
    n_imgs, n_txt = len(all_imgs_embds), len(all_captions_embds)

    all_ids = [x.tolist() for x in all_ids]

    all_embds = all_imgs_embds + all_captions_embds
    all_embds = [x.detach().cpu().numpy() for x in all_embds]
    all_colors = [0]*n_imgs + [1]*n_txt
    all_colors_column = [[x] for x in all_colors]
    features_data = np.hstack([all_embds, all_colors_column])

    N_COMPONENTS = 2
    out_tsne = TSNE(perplexity=15, n_components=N_COMPONENTS, verbose=1).fit_transform(features_data)

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

    plt.savefig(f'./despues2.png', dpi=300, bbox_inches='tight')




def create_and_compute_KNN(fit_features_x, fit_features_y, features_to_predict, n_neighbors=1):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(fit_features_x, fit_features_y)
    pred_labels = knn.predict(features_to_predict)
    pred_labels = np.array(pred_labels)
    return pred_labels


if __name__ == "__main__":
    USE_PICKLE_INFO = False

    if not USE_PICKLE_INFO:
        #LOAD SAVED MODEL
        #========================================================
        WEIGHT_SAVE_PATH = './weights/model_img_task_c_a_1epoch.pth'

        model = Model()

        txt_model = model.model_txt
        img_model = model.model_img

        model.embedding_model.load_state_dict(torch.load(WEIGHT_SAVE_PATH, map_location='cpu'))
        #========================================================

        dataset_train = Dataset(True)
        dataloader_train = DataLoader(dataset_train, drop_last=True, shuffle=False)

        dataset_val = Dataset(False)
        dataloader_val = DataLoader(dataset_val, drop_last=True, shuffle=False)

    else:
        model = None
        dataloader_val = None

    tsne_embeddings(img_model, txt_model, dataloader_val, use_saved_pickles=USE_PICKLE_INFO, title="Embeddings after the alignment")


    imgs_embds_fit, captions_embds_fit, ids_fit = get_embeddings(model, dataloader_train, max_samples=200, save_pickles=False)
    imgs_embds_retrieve, captions_embds_retrieve, ids_retrieve = get_embeddings(model, dataloader_val, max_samples=200, save_pickles=False)

    imgs_embds_fit = [x.detach().cpu().numpy() for x in imgs_embds_fit]

    imgs_embds_retrieve = [x.detach().cpu().numpy() for x in imgs_embds_retrieve]
    captions_embds_fit = [x.detach().cpu().numpy() for x in captions_embds_fit]
    ids_fit = [x.tolist() for x in ids_fit]

    pred_labels = create_and_compute_KNN(np.array(captions_embds_fit), np.array(ids_fit), np.array(imgs_embds_fit), n_neighbors=1)

    print(ids_fit)
    print(pred_labels)

    correct = 0
    total = 0
    for ids in ids_fit:
        for pred in pred_labels:
            if pred == ids:
                correct += 1
            total += 1
    
    print(f"Accuracy: {correct / total}")