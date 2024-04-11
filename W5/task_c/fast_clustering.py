
"""
This is a more complex example on performing clustering on large scale dataset.

This examples find in a large set of sentences local communities, i.e., groups of sentences that are highly
similar. You can freely configure the threshold what is considered as similar. A high threshold will
only find extremely similar sentences, a lower threshold will find more sentence that are less similar.

A second parameter is 'min_community_size': Only communities with at least a certain number of sentences will be returned.

The method for finding the communities is extremely fast, for clustering 50k sentences it requires only 5 seconds (plus embedding comuptation).

In this example, we download a large set of questions from Quora and then find similar questions in this set.
"""

from sentence_transformers import SentenceTransformer, util
import os
import csv
import time
import pickle

from dataloader_original_set import OriginalDataset
import torch
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE


def get_tsne_plot(embeddings, clusters):

    N_COMPONENTS = 2
    out_tsne = TSNE(n_components=N_COMPONENTS, verbose=1).fit_transform(embeddings)


if __name__ == '__main__':
    
    # Model for computing sentence embeddings. We use one trained for similar questions detection
    model = SentenceTransformer("all-MiniLM-L6-v2")

    train_set = OriginalDataset(train=True)

    dataloader_train = DataLoader(train_set, batch_size=16, drop_last=False, shuffle=False)

    all_captions = []
    for _, captions, all_ids in dataloader_train:
        all_captions.extend(captions)

    print("Encoding the captions. This might take a while ...\n")
    corpus_embeddings = model.encode(all_captions, batch_size=25, show_progress_bar=True, convert_to_tensor=True)





    print("Start clustering")
    start_time = time.time()

    # Two parameters to tune:
    # min_cluster_size: Only consider cluster that have at least 25 elements
    # threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
    clusters = util.community_detection(corpus_embeddings, min_community_size=1000, threshold=0.75)

    print("Clustering done after {:.2f} sec".format(time.time() - start_time))

    print("Saving corpus embeddings and clusters pickles")
    with open('corpus_embeddings_1000.pkl', 'wb') as fp:
        pickle.dump(corpus_embeddings, fp)

    with open('clusters_1000.pkl', 'wb') as fp:
        pickle.dump(clusters, fp)


    # Print for all clusters the top 3 and bottom 3 elements
    for i, cluster in enumerate(clusters):
        print("\nCluster {}, #{} Elements ".format(i + 1, len(cluster)))
        for sentence_id in cluster[0:3]:
            print("\t", all_captions[sentence_id])
        print("\t", "...")
        for sentence_id in cluster[-3:]:
            print("\t", all_captions[sentence_id])

        print("\n\n ---------------- \n\n")