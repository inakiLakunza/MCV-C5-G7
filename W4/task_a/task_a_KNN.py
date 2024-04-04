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

# from task_a import Model
from task_b import Model

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
                   max_samples=300,
                   use_saved_pickles = False,
                   wanted_embeds = "task_a",
                   save_pickles=False):
    
    if use_saved_pickles:
        
        PICKLE_ROOT = os.path.join('./', 'pickles', wanted_embeds)

        img_embds_pickle_path = os.path.join(PICKLE_ROOT, 'imgs.pkl')
        caption_embds_pickle_path = os.path.join(PICKLE_ROOT, 'captions.pkl')
        ids_pickle_path = os.path.join(PICKLE_ROOT, 'ids.pkl')

        all_imgs_embds, all_captions_embds, all_ids = use_pickle_info(img_embds_pickle_path, caption_embds_pickle_path, ids_pickle_path)

    else:
        pbar = tqdm(enumerate(dataloader), total=max_samples)
        #pbar = tqdm(enumerate(dataloader), total=math.ceil(len(dataloader) / 1000))

        all_imgs_embds = []
        all_captions_embds = []
        all_ids = []
        all_captions = []

        counter = 0

        # We will use the IDs for the TSNE plot
        for batch_idx, (anchor_img, captions, ids) in pbar:

            if counter > max_samples: break

            # Image Embeddings
            anchor_img = anchor_img.float() # (bs, 3, 224, 224)
            anchor_out = img_model(anchor_img) # (bs, 4096)

            # Text Embeddings
            text_vecs = []
            for caption in captions:
                word_vecs = []
                for word in caption.split():
                    if word.lower() in model.model_txt:
                        word_vecs.append(torch.tensor(model.model_txt[word.lower()]))
                text_vecs.append(torch.stack(word_vecs).mean(dim=0))
            text_vecs = torch.stack(text_vecs).to(model.device)

            pos_embds = txt_model(text_vecs) # (bs, 4096)


            # Add the Batch's imgs, captions and ID's to the full lists
            all_imgs_embds.extend(anchor_out)
            all_captions_embds.extend(pos_embds)
            all_ids.extend(ids)

            counter += 1

        if save_pickles:
            os.makedirs(f"./pickles/{wanted_embeds}", exist_ok=True)

            with open(f"./pickles/{wanted_embeds}/captions.pkl", "wb") as file:
                pickle.dump(all_captions_embds, file)
            with open(f"./pickles/{wanted_embeds}/ids.pkl", "wb") as file:
                pickle.dump(all_ids, file)
            with open(f"./pickles/{wanted_embeds}/imgs.pkl", "wb") as file:
                pickle.dump(all_imgs_embds, file)

    return all_imgs_embds, all_captions_embds, all_ids





def tsne_embeddings(img_model, txt_model, dataloader, title="TSNE_plot_task_a",
                    max_samples=100,
                    use_saved_pickles = False,
                    wanted_embeds = "task_a",
                    save_pickles=False
                    ):

    all_imgs_embds, all_captions_embds, all_ids = get_embeddings(img_model, txt_model, dataloader,
                                                                 max_samples=max_samples,
                                                                 use_saved_pickles=use_saved_pickles,
                                                                 wanted_embeds=wanted_embeds,
                                                                 save_pickles=save_pickles
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

    plt.savefig(f'./despues.png', dpi=300, bbox_inches='tight')


def pk(actual, predicted, k=10):
    """
    Computes the precision at k.
    This function computes the precision at k between the query image and a list
    of database retrieved images.
    Parameters
    ----------
    actual : int
             The element that has to be predicted
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The precision at k over the input
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0
    for i in range(len(predicted)):
        if actual == predicted[i]:
            score += 1
    
    return score / len(predicted)


def mpk(actual, predicted, k=10):
    """
    Computes the precision at k.
    This function computes the mean precision at k between a list of query images and a list
    of database retrieved images.
    Parameters
    ----------
    actual : list
             The query elements that have to be predicted
    predicted : list
                A list of predicted elements (order does matter) for each query element
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The precision at k over the input
    """
    pk_list = []
    for i in range(len(actual)):
        score = pk(actual[i], predicted[i], k)
        pk_list.append(score)
    return np.mean(pk_list)

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


def create_KNN_and_get_predictions(fit_features_x, fit_features_y, features_to_predict, n_neighbors=1):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(fit_features_x, fit_features_y)
    distance, neighbors = knn.kneighbors(features_to_predict, return_distance=True)
    
    return order_neighbors_by_distance(fit_features_y[neighbors], distance)


def compute_mapk_1_5(neighbors, labels):
    mpk1 = mpk(labels, neighbors, 1)
    mpk5 = mpk(labels, neighbors, 5)
    return mpk1, mpk5

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


def show_img_and_related_captions(img, knn, captions_fit, fit_features_y):

    image_tensor = img.clone().detach()

    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()

    image_np = image_tensor.detach().cpu().numpy()

    # Convert from (C, H, W) to (H, W, C)
    image_np = np.transpose(image_np, (1, 2, 0))

    img = img.float() # (bs, 3, 224, 224)
    img_emb = img_model(img) # (bs, 4096)

    distance, neighbors = knn.kneighbors(img_emb.detach().cpu().numpy(), return_distance=True)
    neighbors, distances = order_neighbors_by_distance(fit_features_y[neighbors], distance)

    neighbors = neighbors[0].tolist()
    print(neighbors)
    indexes = [np.where(fit_features_y == x)[0].tolist()[0] for x in neighbors]
    print(indexes)
    captions = [captions_fit[x] for x in indexes]

    print(image_tensor.type)

    plt.imshow(image_np)
    plt.axis('off')
    plt.savefig('image.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f'captions:')
    for caption in captions:
        print(caption)




if __name__ == "__main__":
    USE_PICKLE_INFO = False

    if not USE_PICKLE_INFO:
        #LOAD SAVED MODEL
        #========================================================
        # task_a
        # WEIGHT_SAVE_PATH_TXT = './weights/text_model_task_a_1epoch_embed_1000.pth'
        # WEIGHT_SAVE_PATH_IMG = './weights/image_model_task_a_1epoch_embed_1000.pth'
        # task_b
        WEIGHT_SAVE_PATH_TXT = './weights/text_model_task_b_1epoch_embed_256.pth'
        WEIGHT_SAVE_PATH_IMG = './weights/image_model_task_b_1epoch_embed_256.pth'

        model = Model()

        txt_model = model.text_model
        img_model = model.model_img

        txt_model.load_state_dict(torch.load(WEIGHT_SAVE_PATH_TXT, map_location='cpu'))
        img_model.load_state_dict(torch.load(WEIGHT_SAVE_PATH_IMG, map_location='cpu'))
        #========================================================

        dataset_train = Dataset(True)
        dataloader_train = DataLoader(dataset_train, drop_last=True, shuffle=False)

        dataset_val = Dataset(False)
        dataloader_val = DataLoader(dataset_val, drop_last=True, shuffle=False)

    else:
        model = None
        dataloader_val = None

    # tsne_embeddings(img_model, txt_model, dataloader_val, use_saved_pickles=USE_PICKLE_INFO, title="Embeddings after the alignment")


    imgs_embds_fit, captions_embds_fit, ids_fit = get_embeddings(img_model, txt_model, dataloader_train, max_samples=200, save_pickles=False)
    imgs_embds_retrieve, captions_embds_retrieve, ids_retrieve = get_embeddings(img_model, txt_model, dataloader_val, max_samples=200, save_pickles=False)

    imgs_embds_fit = [x.detach().cpu().numpy() for x in imgs_embds_fit]

    imgs_embds_retrieve = [x.detach().cpu().numpy() for x in imgs_embds_retrieve]
    captions_embds_fit = [x.detach().cpu().numpy() for x in captions_embds_fit]
    ids_fit = [x.tolist() for x in ids_fit]

    sorted_neighbors, knn = get_top_K_neighbors_and_distances(np.array(captions_embds_fit), np.array(ids_fit), np.array(imgs_embds_fit), n_neighbors=5)
    print(compute_mapk_1_5(sorted_neighbors, ids_fit))

    anchor_img, caption, id = dataset_train.get_random_image()
    print(f'Image ID: {id} \n Ground Truth Caption = "{caption}"')
    show_img_and_related_captions(anchor_img, knn, captions_embds_fit, np.array(ids_fit))


    '''
    print(ids_fit)
    print(pred_labels)

    correct = 0
    total = 0
    for i, id in enumerate(ids_retrieve):
        if id in neighbor_labels[i]:
            print(neighbor_labels[i])
            correct += 1
        total += 1
    
    print(f"Top-5 Accuracy: {correct / total}")
    '''