# Code for Discarding Noisy Instance Dynamically 
# With Gaussian Mixture Model

import numpy as np
import math
import scipy.stats as stats
import torch

from sklearn.mixture import GaussianMixture as GMM
# from .util import estimate_purity

__all__=['fit_mixture', 'fit_mixture_bmm']


import datetime
def return_time():
	e = datetime.datetime.now()
	return str(e.hour) + ":" + str(e.minute) + ":" + str(e.second)


# TODO: figure out where you are getting the scores from 
def fit_mixture(scores, labels, p_threshold=0.5):

	clean_labels = []
	indexes = np.array(range(len(scores)))
	
	for cls in np.unique(labels):
	
		# print(return_time())
		
		cls_index = indexes[labels==cls] # FIXME: what the hell does this do 
		
		feats = scores[labels==cls]
		feats_ = np.ravel(feats).astype(np.float).reshape(-1, 1) # TODO: figure out what the output will be
		gmm = GMM(n_components=2, covariance_type='full', tol=1e-3, max_iter=100)
		
		gmm.fit(feats_)
		prob = gmm.predict_proba(feats_)
		prob = prob[:, gmm.means_.argmax()] # TODO: what does this do
		
		clean_labels += [cls_index[clean_idx] for clean_idx in range(len(cls_index)) if prob[clean_idx] > p_threshold]
		
		# print(return_time())

	print(len(clean_labels))

	return np.array(clean_labels, dtype=np.int64)

# TODO: this one is using some kind of beta model 
def fit_mixture_bmm(scores, labels, p_threshold):
    '''
    Assume the distribution of scores: bimodel mixture model

    return clean labels 
    that belongs to the clean cluster by fitting the score distribution to BMM
    '''
    return # FIXME: remove the code later if not req
    clean_labels = []
    indexes = np.array(range(len(scores)))
    for cls in np.unique(labels):
        cls_index = indexes[labels==cls]
        feats = scores[labels==cls]
        feats_ = np.ravel(feats).astype(np.float).reshape(-1, 1)
        feats_ = (feats_ - feats_.min()) / (feats_.max() - feats_.min())
        bmm = BetaMixture(max_iters=100)
        bmm.fit(feats_)
        
        mean_0 = bmm.alphas[0] / (bmm.alphas[0] + bmm.betas[0])
        mean_1 = bmm.alphas[1] / (bmm.alphas[1] + bmm.betas[1])
        clean = 0 if mean_0 > mean_1 else 1
        
        init = bmm.predict(feats_.min(), p_threshold, clean)
        for x in np.linspace(feats_.min(), feats_.max(), 50):
            pred = bmm.predict(x, p_threshold, clean)
            if pred != init:
                bound = x
                break
        
        clean_labels += [cls_index[clean_idx] for clean_idx in range(len(cls_index)) if feats[clean_idx] > bound] 
    
    return np.array(clean_labels, dtype=np.int64)

