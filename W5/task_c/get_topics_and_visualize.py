
import io
import pickle
import torch
from torch.utils.data import DataLoader
from bertopic import BERTopic
from dataloader_original_set import OriginalDataset
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import pandas as pd
import seaborn as sns

os.environ["KERAS_BACKEND"] = "torch"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def plot_tsne(embeds, labels, topics):

    features_data =  np.hstack([embeds, labels])

    N_COMPONENTS = 2
    out_tsne = TSNE(n_components=N_COMPONENTS, verbose=1).fit_transform(features_data)

    df = pd.DataFrame(dict(x=out_tsne[:, 0], y=out_tsne[:, 1], label=labels))

    replace_dict = {}
    for i in range(len(topics)): replace_dict[i] = topics[i]
    df['label'] = df['label'].replace(replace_dict)
    sns.set_style("whitegrid")
    sns.scatterplot(x="x", y="y", hue="label", data=df, legend=False)
    label_positions = df.groupby('label').mean().reset_index()
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., ncol=3)
    for _, row in label_positions.iterrows():
        plt.text(row['x'], row['y'], row['label'], horizontalalignment='center', verticalalignment='center', 
                 fontsize=9, weight='bold', alpha=0.7)
    plt.tight_layout()

    plt.savefig(f'./tsne_with_topics.png', dpi=300, bbox_inches='tight')










if __name__ == '__main__':

    print("EMPIEZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    train_set = OriginalDataset(train=True)

    print("Train set created")

    dataloader_train = DataLoader(train_set, batch_size=16, drop_last=False, shuffle=False)

    print("Dataloader created")

    all_captions = []
    counter = 0
    start_time = time.time()
    for captions, all_ids in dataloader_train:
        if counter == 1000:
            counter = 0
            end_time = time.time()
            print(f"Needed time for loading 1000 batches: {end_time-start_time}")
            start_time = end_time
        else: counter += 1

        all_captions.extend(captions)
    
    # SAVE ALL CAPTIONS
    with open('all_captions.pkl', 'rb') as fp:
        pickle.dump(all_captions, fp)

    
    # print("All captions grouped in a list")

    print("Loading corpus embeddings...")
    with open('corpus_embeddings_1000.pkl', 'rb') as fp:
        if DEVICE == 'cuda': corpus_embeddings = pickle.load(fp)
        else: corpus_embeddings = CPU_Unpickler(fp).load()

    print(corpus_embeddings[:10])


    print("Loading clusters...")
    with open('clusters_1000.pkl', 'rb') as fp:
        if DEVICE == 'cuda': clusters = pickle.load(fp)
        else: clusters = CPU_Unpickler(fp).load()

    print("Clusters loaded")

    topic_model = BERTopic()

    cluster_labels = []
    embeds_by_cluster = []
    cluster_topics = []
    for i, cluster in enumerate(clusters):

        cluster_label_list = [[i]]*len(cluster)
        cluster_labels.extend(cluster_label_list)

        cluster_embds = [corpus_embeddings[j] for j in cluster]
        embeds_by_cluster.extend(cluster_embds)

        caption_list = [all_captions[j] for j in cluster]
        topics, probs = topic_model.fit_transform(caption_list)
        
        # Get 3 most probable topics
        topic_prob_pairs = zip(topics, probs)
        sorted_pairs = sorted(topic_prob_pairs, key=lambda x: x[1], reverse=True)

        top_three = sorted_pairs[:3]
        topic_sentence = ""
        for topic, _ in top_three:
            topic_sentence = str(topic_sentence) + "_" + str(topic)
        topic_sentence = topic_sentence[1:]
        cluster_topics.append(topic_sentence)


    # SAVE CLUSTER TOPICS

    plot_tsne(embeds_by_cluster, cluster_labels, cluster_topics) 
    