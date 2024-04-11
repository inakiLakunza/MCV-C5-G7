

import sys
import os

import torch
from torch.utils.data import DataLoader

import numpy as np
import json
from tqdm import tqdm

from dataloader_original_set import OriginalDataset

from keyphrasetransformer import KeyPhraseTransformer

import pickle

from utils import *

from bertopic import BERTopic



def cluster_of_interest(top_three):
    for topic in top_three:
        s = topic.lower()
        if s in topics_of_interest: return True

    return False


def save_cluster_of_interest(cluster):
    pass




if __name__ == '__main__':

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    topics_of_interest = [
        "horse",
        "zebra",
        "sheep",
        "elephant",
        "cow"
    ]

    test_captions = [
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house.",
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house.",
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house.",
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house.",
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house.",
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house.",
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house.",
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house.",
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house.",
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house.",
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house.",
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house.",
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house.",
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house.",
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house.",
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house.",
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house.",
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house.",
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house.",
        "A brown horse is grazing in the grass.",
        "A skinny horse is grazing in a field.",
        "A brown horse is grazing grass near a red house."
        
    ]

    topic_model = BERTopic()
    topic_ids, probs = topic_model.fit_transform(test_captions)

    for i in range(len(topic_ids)):
        topic_id = topic_ids[i]
        topics = topic_model.get_topic(topic_id)

        first_3 = []
        for topic, _ in topics[:3]: 
            first_3.append(topic)

        print(f"Caption: {test_captions[i]}\n And its 3 most important topics:")
        print(first_3)

        print("\n\n ------------------------\n\n")

    sys.exit()
    
    
    
    print("Loading all captions...")
    with open('all_captions.pkl', 'rb') as fp:
        all_captions = pickle.load(fp)

    print("Loading clusters...")
    with open('clusters_1000.pkl', 'rb') as fp:
        if DEVICE == 'cuda': clusters = pickle.load(fp)
        else: clusters = CPU_Unpickler(fp).load()

    print("Creating train dataloader...")
    train_set = OriginalDataset(train=True)
    dataloader_train = DataLoader(train_set, 
                                  batch_size=16,
                                  drop_last=False,
                                  shuffle=False)

    print("Loading topic model...")
    topic_model = BERTopic()


    for i, cluster in enumerate(clusters):
        print(f"Analyzing cluster number {i}")

        caption_list = [all_captions[j] for j in cluster]
        topics, probs = topic_model.fit_transform(caption_list)
        
        # Get 3 most probable topics
        topic_prob_pairs = zip(topics, probs)
        sorted_pairs = sorted(topic_prob_pairs, key=lambda x: x[1], reverse=True)

        top_three = sorted_pairs[:3]

        if cluster_of_interest(top_three):
            save_cluster_of_interest(cluster)
        



    '''
    print("Loading original dataset...")
    train_set = OriginalDataset(train=True)
    val_set   = OriginalDataset(train=False)

    dataloader_train = DataLoader(train_set, 
                                  batch_size=16,
                                  drop_last=False,
                                  shuffle=False)
    
    dataloader_val   = DataLoader(val_set, 
                                  batch_size=16,
                                  drop_last=False,
                                  shuffle=False)


    print("Loading clusters...")
    with open('clusters.pkl', 'rb') as fp:
        if DEVICE == 'cuda': clusters = pickle.load(fp)
        else: clusters = CPU_Unpickler(fp).load()
    

    # ITERATING THROUGH THE DATALOADER GIVES US THE FOLLOWING OUTPUT:
    # ANCHOR_IMGS (16, 225, 225, 3)  , CAPTIONS (16) , IDS (16)
    #anchor_imgs, captions, ids = next(iter(dataloader_val))
    
    #for batch_idx, (anchor_imgs, captions, ids) in dataloader_train:
    '''


    



    

