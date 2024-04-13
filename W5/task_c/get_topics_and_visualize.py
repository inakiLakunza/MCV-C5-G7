
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


def plot_tsne(embeds, labels, topics, topics_name):

    labels = [x for x in labels]
    embeds = [x.detach().cpu().numpy() for x in embeds]
    topics_column = [[x] for x in topics]
    #topics_column = [[i] for i in range(len(topics))]

    features_data =  np.hstack([embeds, topics_column])

    N_COMPONENTS = 2
    out_tsne = TSNE(n_components=N_COMPONENTS, verbose=1).fit_transform(features_data)
    print('Topics:'+str(len(topics)))
    print('Out_tsne:'+str(len(out_tsne[:, 0])))
    df = pd.DataFrame(dict(x=out_tsne[:, 0], y=out_tsne[:, 1], label=topics))

    replace_dict = {i: topic for i, topic in enumerate(topics_name)}
    df['label'] = df['label'].replace(replace_dict)
    sns.set_style("whitegrid")
    sns.scatterplot(x="x", y="y", hue="label", data=df, legend=False)
    label_positions = df.groupby('label').mean().reset_index()
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., ncol=3)
    for _, row in label_positions.iterrows():
        plt.text(row['x'], row['y'], row['label'], horizontalalignment='center', verticalalignment='center', 
                 fontsize=5, weight='bold', alpha=0.7)
    plt.tight_layout()

    plt.savefig(f'./tsne_with_topics_named_2.png', dpi=300, bbox_inches='tight')










if __name__ == '__main__':

    topics_to_avoid = [
                       "a", "with", "and", "the", "of",
                       "is", "there", "in", "on", "up",
                       "above", "inside", "next", "close",
                       "to", "top", "an", "one", "two",
                       "some", "together", "has", "through",
                       "bunch", "it", "to", "some", "down",
                       "man", "women", "children", "child",
                       "men", "woman", "are", "his", "her",
                       ]

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)
    '''
    train_set = OriginalDataset(train=True)

    print("Train set created")

    dataloader_train = DataLoader(train_set, batch_size=16, drop_last=False, shuffle=False)

    print("Dataloader created")
   
    all_captions = []
    counter = 0
    start_time = time.time()
    for captions, all_ids, _ in dataloader_train:
        if counter == 1000:
            counter = 0
            end_time = time.time()
            print(f"Needed time for loading 1000 batches: {end_time-start_time}")
            start_time = end_time
        else: counter += 1

        all_captions.extend(captions)
    
    # SAVE ALL CAPTIONS
    with open('all_captions.pkl', 'wb') as fp:
        pickle.dump(all_captions, fp)

    '''

    print("Loading captions...")
    with open('pkl_all_captions.pkl', 'rb') as fp:
        if DEVICE == 'cuda': all_captions = pickle.load(fp)
        else: all_captions = CPU_Unpickler(fp).load()

    with open('pkl_all_ids.pkl', 'rb') as fp:
        if DEVICE == 'cuda': all_ids = pickle.load(fp)
        else: all_ids = CPU_Unpickler(fp).load()

    with open('pkl_all_img_ids.pkl', 'rb') as fp:
        if DEVICE == 'cuda': all_img_ids = pickle.load(fp)
        else: all_img_ids = CPU_Unpickler(fp).load()

    print("Captions loaded")

    print("All captions grouped in a list")

    print("Loading corpus embeddings...")
    with open('corpus_embeddings_1000.pkl', 'rb') as fp:
        if DEVICE == 'cuda': corpus_embeddings = pickle.load(fp)
        else: corpus_embeddings = CPU_Unpickler(fp).load()

    #print(corpus_embeddings[:10])


    print("Loading clusters...")
    with open('clusters_1000.pkl', 'rb') as fp:
        if DEVICE == 'cuda': clusters = pickle.load(fp)
        else: clusters = CPU_Unpickler(fp).load()

    print("Clusters loaded")

    topic_model = BERTopic()

    #topics, probs = topic_model.fit_transform(all_captions)
    #print("fit transform computed...")

    # with open('topics.pkl', 'rb') as fp:
    #     topics = pickle.load(fp)
        
    # with open('probs.pkl', 'rb') as fp:
    #     probs = pickle.load(fp)


    # with open('topic_model.pkl', 'rb') as fp:
    #     topic_model = pickle.load(fp)

    # print("Pickles loaded")


    cluster_labels = []
    embeds_by_cluster = []
    cluster_topics_list = []
    topics_name = []
    cont=0
    for i, cluster in enumerate(clusters):
        print(f"Analyzing cluster {i}")

        cluster_label_list = [[i]]*len(cluster)
        cluster_labels.extend(cluster_label_list)

        cluster_embds = [corpus_embeddings[j] for j in cluster]
        embeds_by_cluster.extend(cluster_embds)

        cluster_captions = [all_captions[j] for j in cluster]
        
        cluster_topics, cluster_probs = topic_model.fit_transform(cluster_captions)

        print("cluster fit transform completed")

        freq_dict = {}
        for topic_idx in cluster_topics:
            topic_list = topic_model.get_topic(topic_idx)
            counter = 0
            for t, p in topic_list:
                t_lower = t.lower()
                if t_lower in topics_to_avoid: continue
                freq_dict[t_lower] = freq_dict.get(t_lower, 0) + 1
                counter+=1
                if counter==3: break

        sorted_freqs = sorted(freq_dict, key=lambda key: freq_dict[key], reverse=True)
        cluster_sentence = "_".join(sorted_freqs[:3])
        print(f"\n\ncluster sentence: {cluster_sentence}\n\n")

        # # Get 3 most probable topics
        # for i in range(len(cluster)):
        #     index = cluster[i]
        #     caption = all_captions[cluster[i]]
        #     caption_in_list_bracket = [caption]
        #     print(topic_model.get_document_info([all_captions[cluster[i]]])["topic"])
        #     print(topic_model.get_topic(topic_model.get_document_info([all_captions[cluster[i]]])["topic"]))
        #     topics_list = topic_model.get_topic(topic_model.get_document_info([all_captions[cluster[i]]])["topic"])
        #     #print(topics_list)
        #     first_3 = []
        #     for topic, _ in topics_list[:3]:
        #         freq_dict[topic] = freq_dict.get(topic, 0) + 1

        # sorted_freqs = sorted(freq_dict, key=lambda key: freq_dict[key], reverse=True)
        # cluster_sentence = "_".join(sorted_freqs[:3])
        # print(cluster_sentence)

        topics_name.append(cluster_sentence)
        for i in range(len(cluster)):
            cluster_topics_list.append(cont)    
        cont=cont+1

    # SAVE CLUSTER TOPICS
    print('Topics-names: ')
    print(topics_name)
    print('Topics: ')
    print(cluster_topics)
    plot_tsne(embeds_by_cluster, cluster_labels, cluster_topics_list, topics_name) 
    