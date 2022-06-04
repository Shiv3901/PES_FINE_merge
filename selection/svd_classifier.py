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

    print(features.shape)

    # features = features.reshape(-1, 32*32*3)

    singular_vector_dict = {}
    with tqdm(total=len(np.unique(labels))) as pbar:
        for index in np.unique(labels):
            # _, s, v = np.linalg.svd(features[labels==index])
            
            # Shiv's code here
            u, s, v = np.linalg.svd(features[labels==index], full_matrices=False)

            singular_vector_dict[index] = u[0]
            pbar.update(1)

    return singular_vector_dict

# TODO: pretty sure that we do not need this function for now (was taken from the FINE paper)
def get_features(model, dataloader):
    return 

def get_score(singular_vector_dict, features, labels, normalization=True):
    
    if normalization:
        # scores = [np.abs(np.inner(singular_vector_dict[labels[indx]], feat/np.linalg.norm(feat))) for indx, feat in enumerate(tqdm(features))]

        scores = []
        for idx, feat in enumerate(tqdm(features)):

            a = singular_vector_dict[labels[idx]]
            b = feat / np.linalg.norm(feat)
            
            tempAns = np.abs(np.inner(a, b.reshape(-1, 32*32*3)))
            print("TempANs Shape: ", tempAns.shape)
            scores.append(tempAns[0])


    else:
        scores = [np.abs(np.inner(singular_vector_dict[labels[indx]], feat)) for indx, feat in enumerate(tqdm(features))]    
    

    return np.array(scores)

# function that fits the labels using GMM 
def fit_mixture(scores, labels, p_threshold=0.30):

    clean_labels = []
    indexes = np.array(range(len(scores)))
    probs = {}

    for idx, cls in enumerate(np.unique(labels)):
        
        cls_index = indexes[labels==cls]
        feats = scores[labels==cls]
        
        gmm = GaussianMixture(n_components=2, covariance_type='diag', tol=1e-6, max_iter=100)

        gmm.fit(feats)
        prob = gmm.predict_proba(feats)
        prob = prob[:, gmm.means_.argmax()]

        for i in range(len(cls_index)):
            probs[cls_index[i]] = prob[i]

        clean_labels += [cls_index[clean_idx] for clean_idx in range(len(cls_index)) if prob[clean_idx] > p_threshold]

    return np.array(clean_labels, dtype=np.int64), probs

from sklearn.decomposition import PCA

def get_score_shiv(current_features):

    pca = PCA(n_components=2, svd_solver='arpack')
    
    return pca.fit_transform(current_features.reshape(-1, 3072))

def get_singular_vector_shiv(features, labels):

    singular_vector_dict = {}
    with tqdm(total=len(np.unique(labels))) as pbar:
        for index in np.unique(labels):

            feats = features[labels==index]
            
            pca = PCA(n_components=2, svd_solver='full', random_state=68)

            _ = pca.fit_transform(feats.reshape(-1, 3072))

            print("Eigen Vector: ", pca.components_[0].shape)
            singular_vector_dict[index] = pca.components_[0][0]

            pbar.update(1)

    return singular_vector_dict


def get_score_individual(features, labels):

    scores_dict = {}
    indexes = np.array(range(len(labels)))

    with tqdm(total=len(np.unique(labels))) as pbar:
        for index in np.unique(labels):
            cls_index = indexes[labels==index]
            feats = features[labels==index]
            
            pca = PCA(n_components=2, svd_solver='full', random_state=68)

            score_vals = pca.fit_transform(feats.reshape(-1, 3072))

            compo = pca.components_

            print(compo.shape)

            for i in range(len(cls_index)):
                scores_dict[cls_index[i]] = score_vals[i]
            pbar.update(1)

    scores = []
    for i in range(len(labels)):
        scores.append(scores_dict[i])
            
    return np.array(scores, dtype=np.double)

def fine(current_features, current_labels, fit='kmeans', previous_features=None, previous_labels=None, p_threshold=0.7):

    # if not previous_features and not previous_labels:
    #     singular_vector_dict = get_singular_vector(previous_features, previous_labels)
    # else:

    # singular_vector_dict = get_singular_vector(current_features, current_labels)

    # scores = get_score(singular_vector_dict, features=current_features, labels=current_labels)
    # print("Scores dimension: ", scores.shape)
    # scores_1 = get_score_shiv(current_features)

    singular_vector_dict = get_singular_vector_shiv(current_features, current_labels)
    scores = get_score(singular_vector_dict, features=current_features, labels=current_labels)
    print(scores.shape)

    # return 

    if 'kmeans' in fit:
        clean_labels = cleansing(scores, current_labels)
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

    for cls in np.unique(labels):
        cls_index = indexes[labels == cls]
        kmeans = cluster.KMeans(n_clusters=2, random_state=0)
        
        # # print("Doing it for: " + str(cls))

        feats = scores[cls_index]

        # # print(feats.shape)
        
        # FIXME: remove this once not needed 
        # if feats.shape[0] < 50: continue

        # feats_ = feats.reshape(feats.shape[0], 32*32*32*3)

        # # print(# print_current_time("start: "))

        labels_ = kmeans.fit(feats).labels_

        # # print(# print_current_time("end: "))

        if np.mean(feats[labels_ == 0]) < np.mean(feats[labels_ == 1]):
            labels_ = 1 - labels_

        # counter = 0

        # for i in labels_:
        #     if i == 0:
        #         counter += 1

        # # print(counter, labels_.shape[0] - counter)
        # # print(np.mean(feats[labels_ == 0]))

        for idx, label in enumerate(labels_):
            if label == 0:
                clean_labels.append(cls_index[idx])

        # clean_labels += cls_index[labels_ == 0].tolist()

    # print("Kmeans: ", # print_current_time("end: "))

    # # print("Inside the fine function: ", len(clean_labels))
        
    return np.array(clean_labels, dtype=np.int64) 

