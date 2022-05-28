import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random 
import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn import cluster
import numpy as np
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

SEED = 110
GPU_ID = 1

torch.cuda.set_device(GPU_ID)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def get_singular_vector(features, labels):

    singular_vector_dict = {}
    with tqdm(total=len(np.unique(labels))) as pbar:
        for index in np.unique(labels):
            _, _, v = np.linalg.svd(features[labels==index])
            singular_vector_dict[index] = v[0]
            pbar.update(1)

    return singular_vector_dict

# TODO: pretty sure that we do not need this function for now
def get_features(model, dataloader):
    return 

def get_score(singular_vector_dict, features, labels, normalization=True):
    
    if normalization:
        scores = [np.abs(np.inner(singular_vector_dict[labels[indx]], feat/np.linalg.norm(feat))) for indx, feat in enumerate(tqdm(features))]
    else:
        scores = [np.abs(np.inner(singular_vector_dict[labels[indx]], feat)) for indx, feat in enumerate(tqdm(features))]    

    return np.array(scores)

# function that fits the labels using GMM 
def fit_mixture(scores, labels, p_threshold=0.50):

    clean_labels = []
    indexes = np.array(range(len(scores)))
    probs = {}

    # print(scores.shape, "Just looking")

    for idx, cls in enumerate(np.unique(labels)):
        cls_index = indexes[labels==cls]
        feats = scores[labels==cls]
        feats_ = np.ravel(feats).astype(np.float).reshape(-1, 1)

        print(print_current_time("Label " + str(idx) + ": "))

        gmm = GaussianMixture(n_components=2, covariance_type='diag', tol=1e-6, max_iter=2)

        gmm.fit(feats_)
        prob = gmm.predict_proba(feats_)
        prob = prob[:, gmm.means_.argmax()]

        for i in range(len(cls_index)):
            probs[cls_index[i]] = prob[i]

        clean_labels += [cls_index[clean_idx] for clean_idx in range(len(cls_index)) if prob[clean_idx] > p_threshold]

    return np.array(clean_labels, dtype=np.int64), probs

from sklearn.decomposition import PCA

def get_score_shiv(current_features):

    print(print_current_time("Start of PCA: "))
    pca = PCA(n_components=1, svd_solver='arpack')
    print(print_current_time("End of PCA: "))

    return pca.fit_transform(current_features.reshape(-1, 3072))

    return pca.get_covariance()

def fine(current_features, current_labels, fit='kmeans', previous_features=None, previous_labels=None, p_threshold=0.7):

    # if not previous_features and not previous_labels:
    #     singular_vector_dict = get_singular_vector(previous_features, previous_labels)
    # else:
    # singular_vector_dict = get_singular_vector(current_features, current_labels)

    # scores = get_score(singular_vector_dict, features=current_features, labels=current_labels)

    scores_1 = get_score_shiv(current_features)

    print("Score 1", scores_1.shape)
    # print("Score 2", scores_2.shape)

    # print(singular_vector_dict)

    # print("Score 1: ", scores_1)

    # print("Score 2: ", scores_2)

    # return None, None

    if 'kmeans' in fit:
        clean_labels = cleansing(scores_1, current_labels)
        probs = None
    elif 'gmm' in fit:
        clean_labels, probs = fit_mixture(scores, current_labels, p_threshold)
    else:
        raise NotImplemented
    
    return clean_labels, probs

from datetime import datetime

def print_current_time(place_holder=""): 
    now = datetime.now()
    return place_holder + now.strftime("%H:%M:%S")

def cleansing(scores, labels):

    indexes = np.array(range(len(scores)))
    clean_labels = []

    print("Kmeans function called here")

    print("labels: ", np.unique(labels))

    # print(scores)

    print(print_current_time("start: "))

    for cls in np.unique(labels):
        cls_index = indexes[labels == cls]
        kmeans = cluster.KMeans(n_clusters=2, random_state=0)
        
        # print("Doing it for: " + str(cls))

        feats = scores[cls_index]

        # print(feats.shape)
        
        # FIXME: remove this once not needed 
        # if feats.shape[0] < 50: continue

        # feats_ = feats.reshape(feats.shape[0], 32*32*32*3)

        # print(print_current_time("start: "))

        labels_ = kmeans.fit(feats).labels_
        
        counter = 0

        for i in labels_:
            if i == 0:
                counter += 1

        print(counter, labels_.shape[0] - counter)

        # print(print_current_time("end: "))

        if np.mean(feats[labels_ == 0]) < np.mean(feats[labels_ == 1]):
            labels_ = 1 - labels_

        clean_labels += cls_index[labels_ == 0].tolist()

    print(print_current_time("end: "))
        
    return np.array(clean_labels, dtype=np.int64) 

