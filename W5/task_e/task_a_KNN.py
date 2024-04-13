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
from uuid import uuid4
import random
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE

from fine_tune import ImageEmbeddingLayer, TextEmbeddingLayer, Model

from get_augmented_2_1 import Augmented_2_1
from get_augmented_turbo import Augmented_Turbo

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
        all_captions = []
        all_images = []

        counter = 0

        # We will use the IDs for the TSNE plot
        for batch_idx, (anchor_img, captions, ids) in pbar:
            all_captions.extend(captions)
            if counter > max_samples: break
            all_images.extend(anchor_img)

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
            pos_embds = txt_model(sentence_embedding) # (bs, 4096)
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

    return all_imgs_embds, all_captions_embds, all_ids, all_captions, all_images





def tsne_embeddings(img_model, txt_model, dataloader, title="TSNE_plot_task_c_a",
                    max_samples=100,
                    use_saved_pickles = False,
                    wanted_embeds = "task_c_a",
                    save_pickles=True,
                    pickle_suffix="",
                    img_name="try_img"
                    ):

    plt.clf()
    all_imgs_embds, all_captions_embds, all_ids, _ , __= get_embeddings(img_model, txt_model, dataloader,
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

    plt.savefig(f'./{img_name}.png', dpi=300, bbox_inches='tight')


def get_top_K_neighbors_and_distances(fit_features_x, fit_features_y, features_to_predict, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(fit_features_x, fit_features_y)
    distances, indices = knn.kneighbors(features_to_predict, return_distance=True)
    sorted_neighbors = []
    for idx, dist in zip(indices, distances):
        labels = fit_features_y[idx]
        neighbor_pairs = sorted(zip(labels, dist), key=lambda x: x[1])
        sorted_neighbors.append(labels)
    return sorted_neighbors, knn


def create_and_compute_KNN(fit_features_x, fit_features_y, features_to_predict, n_neighbors=1):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(fit_features_x, fit_features_y)
    pred_labels = knn.predict(features_to_predict)
    pred_labels = np.array(pred_labels)
    return pred_labels

# THANKS ChatGPT
def order_neighbors_by_distance(neighbors, distances):
    # Combine neighbors and distances into a list of tuples
    neighbor_distance_pairs = list(zip(neighbors, distances))
    # Sort the list based on distances (the second element of each tuple)
    sorted_pairs = sorted(neighbor_distance_pairs, key=lambda x: np.linalg.norm(x[1]))
    # Unpack the sorted tuples into separate lists
    sorted_neighbors = [pair[0] for pair in sorted_pairs]
    sorted_distances = [pair[1] for pair in sorted_pairs]
    return sorted_neighbors, sorted_distances



def show_img_and_related_captions(img, knn, captions_fit, fit_features_y, gt_caption, image_id):

    image_tensor = img.clone().detach()

    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()

    image_np = image_tensor.detach().cpu().numpy()

    image_np = np.transpose(image_np, (1, 2, 0))

    img = img.float() # (bs, 3, 224, 224)
    img_emb = img_model(img) # (bs, 4096)

    distance, neighbors = knn.kneighbors(img_emb.detach().cpu().numpy(), return_distance=True)
    neighbors, distances = order_neighbors_by_distance(fit_features_y[neighbors], distance)

    neighbors = neighbors[0].tolist()
    indexes = [np.where(fit_features_y == x)[0].tolist()[0] for x in neighbors]

    captions = [gt_caption] + [captions_fit[x] for x in indexes]

    os.makedirs('./results_task_a_prueba/', exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Image
    img_ax = fig.add_axes([0.05, 0.1, 0.4, 0.8])  # Increased left padding
    img_ax.imshow(image_np)
    img_ax.set_title(f"Image with ID {image_id}", fontsize=16)  # Increased title font size
    img_ax.axis('off')

    gt_title_ax = fig.add_axes([0.55, 0.85, 0.4, 0.05])
    gt_title_ax.text(0.5, 0.5, "Ground Truth Caption", va='center', ha='center', fontsize=14)
    gt_title_ax.axis('off')

    gt_cap_ax = fig.add_axes([0.55, 0.75, 0.4, 0.1])
    gt_cap_ax.text(0.5, 0.5, gt_caption, va='center', ha='center', fontsize=14, wrap=True, 
                   bbox=dict(facecolor='green', alpha=0.5))
    gt_cap_ax.axis('off')

    ax.plot([0.55, 0.95], [0.73, 0.73], color="black", linestyle="--", transform=fig.transFigure, clip_on=False)

    pred_title_ax = fig.add_axes([0.55, 0.65, 0.4, 0.05])
    pred_title_ax.text(0.5, 0.5, "Predicted Captions", va='center', ha='center', fontsize=14)
    pred_title_ax.axis('off')

    for i, caption in enumerate(captions[1:], start=1):
        pred_cap_ax = fig.add_axes([0.55, 0.65 - i*0.13, 0.4, 0.12])
        pred_cap_ax.text(0.5, 0.5, caption, va='center', ha='center', fontsize=12, wrap=True,
                         bbox=dict(facecolor='lightgray', alpha=0.5))
        pred_cap_ax.axis('off')

    pid = str(uuid4())
    plt.savefig(os.path.join(f"./results_task_a_prueba/{pid}.png"), bbox_inches='tight', pad_inches=0.5)

    print(f'captions:')
    for caption in captions:
        print(caption)

    print(f"Saved on {os.path.join(f'./results_task_a_prueba/{pid}.png')}")





if __name__ == "__main__":
    USE_PICKLE_INFO = False

    if not USE_PICKLE_INFO:
        #LOAD SAVED MODEL
        #========================================================
        WEIGHT_SAVE_PATH_TXT = '/ghome/group07/C5-W4/task_c/weights/text_model_task_a_1epoch.pth'
        WEIGHT_SAVE_PATH_IMG = '/ghome/group07/C5-W4/task_c/weights/image_model_task_a_1epoch.pth'
        
        #model_no_train = Model()
        #txt_model_no_train = model_no_train.text_model
        #img_model_no_train = model_no_train.model_img

        model = Model(WEIGHT_SAVE_PATH_TXT,WEIGHT_SAVE_PATH_IMG)

        txt_model = model.text_model
        img_model = model.model_img

        txt_model.load_state_dict(torch.load(WEIGHT_SAVE_PATH_TXT, map_location=torch.device('cpu')))
        img_model.load_state_dict(torch.load(WEIGHT_SAVE_PATH_IMG, map_location=torch.device('cpu')))
        #========================================================

        dataset_train = Dataset(True)
        # append image we want
        dataset_train.append_known_image(
            image_id='19239',
            id=526692,
            caption='a couple zebra drinking water out in the desert',
            index=499
        )
        dataloader_train = DataLoader(dataset_train, drop_last=True, shuffle=False)

        dataset_val = Dataset(False)
        dataloader_val = DataLoader(dataset_val, drop_last=True, shuffle=False)

    else:
        model = None
        dataloader_val = None

    #tsne_embeddings(img_model_no_train, txt_model_no_train, dataloader_val, use_saved_pickles=USE_PICKLE_INFO, title="Embeddings before the alignment", img_name="before_c_a")
    #tsne_embeddings(img_model, txt_model, dataloader_val, use_saved_pickles=USE_PICKLE_INFO, title="Embeddings after the alignment", img_name="after_c_a")

    imgs_embds_fit, captions_embds_fit, ids_fit, captions_train,imgs_fit = get_embeddings(img_model, txt_model, dataloader_train, max_samples=500, save_pickles=False)
    imgs_embds_retrieve, captions_embds_retrieve, ids_retrieve, _, imgs_retrieve = get_embeddings(img_model, txt_model, dataloader_val, max_samples=500, save_pickles=False)

    imgs_embds_fit = [x.detach().cpu().numpy() for x in imgs_embds_fit]

    imgs_embds_retrieve = [x.detach().cpu().numpy() for x in imgs_embds_retrieve]
    captions_embds_fit = [x.detach().cpu().numpy() for x in captions_embds_fit]
    ids_fit = [x.tolist() for x in ids_fit]

    sorted_neighbors, knn = get_top_K_neighbors_and_distances(np.array(captions_embds_fit), np.array(ids_fit), np.array(imgs_embds_fit), n_neighbors=5)
    pred_labels = create_and_compute_KNN(np.array(captions_embds_fit), np.array(ids_fit), np.array(imgs_embds_fit), n_neighbors=1)

    n_examples = 1
    for n in range(n_examples):
        i = random.randint(0, 499)
        i = 499
        print(imgs_fit[i])
        print(f'Image ID: {ids_fit[i]} \n Ground Truth Caption = "{captions_train[i]}"')
        show_img_and_related_captions(imgs_fit[i], knn, captions_train, np.array(ids_fit), captions_train[i], ids_fit[i])
    

    # print(ids_fit)
    # print(pred_labels)

    # correct = 0
    # total = 0
    # for ids in ids_fit:
    #     for pred in pred_labels:
    #         if pred == ids:
    #             correct += 1
    #         total += 1
    
    # print(f"Accuracy: {correct / total}")