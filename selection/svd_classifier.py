import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random 
import os
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn import cluster
import numpy as np
import warnings
from tqdm import tqdm
import torchvision

from PIL import Image

warnings.filterwarnings("ignore")

SEED = 110
GPU_ID = 1

ARGS = None

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

# transform = transforms.Compose([
#     transforms.PILToTensor()
# ])

def get_features_custom(model, data):

    for i, val in enumerate(data):
        # val = torch.tensor(val)
        image_from_array = torchvision.transforms.functional.to_pil_image(val)
        img_tensor = torchvision.transforms.functional.to_tensor(image_from_array)
        # print(img_tensor.size())
        input = model(img_tensor)
        input = input.cuda

        feature = model.forward(input, lout=4)
        feature = F.avg_pool2d(feature, 4)
        feature = feature.view(feature.size(0), -1)

        if i == 0:
            features = feature.detach().cpu()
        else:
            features = np.concatenate((features, feature.detach().cpu()), axis=0)

    return features

# TODO: pretty sure that we do not need this function for now (was taken from the FINE paper)
def get_features(model, dataloader):

    labels = np.empty((0,))

    model.eval()
    model.cuda()

    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        input, label = data
        input, label = input.cuda(), label.long()

        # print(input.size())

        feature = model.forward(input, lout=4)
        feature = F.avg_pool2d(feature, 4)
        feature = feature.view(feature.size(0), -1)

        labels = np.concatenate((labels, label.cpu()))
        if i == 0:
            features = feature.detach().cpu()
        else:
            features = np.concatenate((features, feature.detach().cpu()), axis=0)
            
    return features, labels 

def get_score(singular_vector_dict, features, labels, normalization=True):
    
    if normalization:
        scores = [[np.abs(np.inner(singular_vector_dict[labels[indx]], feat/np.linalg.norm(feat)))] for indx, feat in enumerate(tqdm(features))]
    else:
        scores = [[np.abs(np.inner(singular_vector_dict[labels[indx]], feat))] for indx, feat in enumerate(tqdm(features))]
        
    return np.array(scores)

'''
parameters = {'knn__n_neighbors':[3,5,10,20,30,50,100], 'knn__weights':['uniform','distance'], 
        'knn__leaf_size': list(range(1, 10)), 'knn__p':[1, 2]}
    gridcvKnn = GridSearchCV(knn_pl, parameters, cv=10, scoring='roc_auc')
    gridcvKnn.fit(x_train, y_train)
    print(f'Best score is {-gridcvKnn.best_score_} for best params of {gridcvKnn.best_params_}\n')

'''

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer


def custom_scorer_function(yt,yp):

    counter = 0

    for i in range(len(yt)):
        if yp[i] == yt[i]:
            counter += 1

    return counter if counter > (len(yp)/ 2) else (len(yp) - counter)

custom_scorer = make_scorer(custom_scorer_function, greater_is_better=True)


# function that fits the labels using GMM 
def fit_mixture(scores, labels, p_threshold=0.2, true_labels=None):

    preds = np.copy(labels)
    clean_labels = []
    indexes = np.array(range(len(scores)))
    for cls in np.unique(labels):
        cls_index = indexes[labels==cls]
        feats = scores[labels==cls]
        
        feats_ = np.ravel(feats).astype(np.float).reshape(-1, 1)
        gmm = None

        # print("GMM Model used of type: " + str(ARGS.hyper_parameter_type))

        if ARGS.hyper_parameter_type == 1:
            gmm = GMM(n_components=2, covariance_type='tied', tol=1e-7, n_init=1,  max_iter=100, random_state=0, warm_start=False)

        elif ARGS.hyper_parameter_type == 2:
            gmm = GMM(n_components=2, covariance_type='diag', tol=1e-7, n_init=1,  max_iter=100, random_state=0, warm_start=False)

        elif ARGS.hyper_parameter_type == 3:
            gmm = GMM(n_components=2, covariance_type='full', tol=1e-7, n_init=1,  max_iter=100, random_state=0, warm_start=False)

        elif ARGS.hyper_parameter_type == 4:
            gmm = GMM(n_components=2, covariance_type='tied', tol=1e-7, n_init=1,  max_iter=100, random_state=0, warm_start=True)

        elif ARGS.hyper_parameter_type == 5:
            gmm = GMM(n_components=2, covariance_type='tied', tol=1e-5, n_init=1,  max_iter=100, random_state=0, warm_start=False)

        elif ARGS.hyper_parameter_type == 6:
            gmm = GMM(n_components=2, covariance_type='tied', tol=1e-7, n_init=1,  max_iter=200, random_state=0, warm_start=False)

        elif ARGS.hyper_parameter_type == 7:
            gmm = GMM(n_components=2, covariance_type='tied', tol=1e-7, n_init=1,  max_iter=300, random_state=0, warm_start=False)

        elif ARGS.hyper_parameter_type == 8:
            gmm = GMM(n_components=2, covariance_type='tied', tol=1e-6, n_init=1,  max_iter=100, random_state=0, warm_start=False)
            
        # true_idxs = true_labels[cls_index]

        # for i in range(len(true_idxs)):
        #     if true_idxs[i] == cls:
        #         true_idxs[i] = 1
        #     else:
        #         true_idxs[i] = 0


        # parameters = {'n_components': [2], 'n_init': [1, 2, 3, 4], 'tol': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3], 'covariance_type': ['full','tied','diag','spherical']}

        # gridcvKnn = GridSearchCV(gmm, parameters, cv=10, scoring=custom_scorer)
        # gridcvKnn.fit(feats_, true_idxs)

        # print("_" * 80)

        # print("Best score: ")

        # print(gridcvKnn.best_score_)
        # print(gridcvKnn.best_params_)

        # print("_" * 80)

        gmm.fit(feats_)
        prob = gmm.predict_proba(feats_)
        prob = prob[:,gmm.means_.argmax()]
        clean_labels_arr = [cls_index[clean_idx] for clean_idx in range(len(cls_index)) if prob[clean_idx] > p_threshold] 
        clean_labels += clean_labels_arr
        
        for idx in clean_labels_arr:
            preds[idx] = cls

    return np.array(clean_labels, dtype=np.int64), np.array(preds, dtype=np.int64)

def fine(current_features, current_labels, fit='kmeans', true_labels=None, prev_features=None, prev_labels=None, p_threshold=0.5, norm=True, eigen=True, args=None):

    ARGS = args

    if eigen is True:
        if prev_features is not None and prev_labels is not None:
            vector_dict = get_singular_vector(prev_features, prev_labels)
        else:
            vector_dict = get_singular_vector(current_features, current_labels)
    # else:
    #     if prev_features is not None and prev_labels is not None:
    #         vector_dict = get_mean_vector(prev_features, prev_labels)
    #     else:
    #         vector_dict = get_mean_vector(current_features, current_labels)

    scores = get_score(vector_dict, features = current_features, labels = current_labels, normalization=norm)
    
    if 'kmeans' in fit:
        clean_labels, preds = cleansing(scores, current_labels)
    elif 'gmm' in fit:
        clean_labels, preds = fit_mixture(scores, current_labels, p_threshold=p_threshold, true_labels=true_labels)
    # elif 'bmm' in fit:
    #     clean_labels = fit_mixture_bmm(scores, current_labels)
    else:
        raise NotImplemented
    
    return clean_labels, preds

from datetime import datetime

def print_current_time(place_holder=""): 
    now = datetime.now()
    return place_holder + now.strftime("%H:%M:%S")

def cleansing(scores, labels):

    preds = np.copy(labels)
    indexes = np.array(range(len(scores)))
    clean_labels = []
    for cls in np.unique(labels):
        cls_index = indexes[labels==cls]
        kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(scores[cls_index].reshape(-1, 1))
        if np.mean(scores[cls_index][kmeans.labels_==0]) < np.mean(scores[cls_index][kmeans.labels_==1]): kmeans.labels_ = 1 - kmeans.labels_
            
        clean_label_arr = cls_index[kmeans.labels_ == 0].tolist()
        clean_labels += clean_label_arr

        for idx in clean_label_arr:
            preds[idx] = cls

    return np.array(clean_labels, dtype=np.int64), np.array(preds, dtype=np.int64)
        

