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
def fit_mixture(scores, labels, p_threshold=0.5):

    clean_labels = []
    indexes = np.array(range(len(scores)))
    probs = {}

    for cls in np.unique(labels):
        cls_index = indexes[labels==cls]
        feats = scores[labels==cls]
        feats_ = np.ravel(feats).astype(np.float).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=10)

        gmm.fit(feats_)
        prob = gmm.predict_proba(feats_)
        prob = prob[:, gmm.means_.argmax()]
        for i in range(len(cls_index)):
            probs[cls_index[i]] = prob[i]

        clean_labels += [cls_index[clean_idx] for clean_idx in range(len(cls_index))]

    return np.array(clean_labels, dtype=np.int64), probs

def fine(current_features, current_labels, fit='kmeans', previous_features=None, previous_labels=None, p_threshold=0.7):

    # if not previous_features and not previous_labels:
    #     singular_vector_dict = get_singular_vector(previous_features, previous_labels)
    # else:
    singular_vector_dict = get_singular_vector(current_features, current_labels)

    scores = get_score(singular_vector_dict, features=current_features, labels=current_labels)

    print("Scores received from get_score function")

    if 'kmeans' in fit:
        clean_labels = cleansing(scores, current_labels)
        probs = None
    elif 'gmm' in fit:
        clean_labels, probs = fit_mixture(scores, current_labels, p_threshold)
    else:
        raise NotImplemented
    
    return clean_labels, probs

from datetime import datetime

def print_current_time(): 
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))
    return

def cleansing(scores, labels):

    indexes = np.array(range(len(scores)))
    clean_labels = []

    for cls in np.unique(labels):
        cls_index = indexes[labels==cls]

        print_current_time()

        kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(scores[cls_index].reshape(-1, 1))
        
        print_current_time()

        if np.mean(scores[cls_index][kmeans.labels_==0]) < np.mean(scores[cls_index][kmeans.labels_==1]): 
            kmeans.labels_ = 1 - kmeans.labels_

        clean_labels += cls_index[kmeans.labels_ == 0].tolist()

    return np.array(clean_labels, dtype=np.int64) 
