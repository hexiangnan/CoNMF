"""
Created on May 6, 2013
Last Update on Oct 9, 2014

The implementation of NMF (non-negative matrix factorization).

Codes adopted from nimfa include:
    euclidean_update();
    _adjustment();

@author: HeXiangnan (xiangnan@comp.nus.edu.sg)    
"""

import data_load as dl
import metrics
import numpy as np
import scipy.sparse as sp
from kmeans import kmeans
from operator import div
from utils import dot,multiply,elop,initialize_random,labels_to_matrix
from scipy.sparse.linalg import svds

def nmf(data, k, norm = "l2", seed="random", post = "direct", gt=None):
    """
    NMF with Euclidean distance as the cost function.
    For comments on input parameters, please refer to conmf.conmf().
    """
    data_norm = dl.norm_data(data, "l2")
    
    # print "Running NMF on a matrix with size ",data.shape
    #nmf_model = nimfa.mf(data_norm, method = "nmf", max_iter = 200, min_residuals = 0.001,n_run =1, rank = k, update = 'euclidean', objective = 'div')
    W,H = factorize(data_norm, seed, post , norm, gt, k) #W is m*k, H is k*n
    
    targets = dl.get_targets(W.T,post) # clustering results. 
    
    return targets,W,H

def factorize(data, seed, post, norm, gt, rank, max_iter=200):
    """
    The factorization of NMF, data = W*H. 
    The input gt (groundtruth) is only for monitoring performance of each iteration.
    
    Note: since calculating the cost function is too slow, we can only use the number of iteration as the stopping critera for efficiency issue. 
    Return: W (m*k) and H (k*n) matrix.
    """   
    V = data
    W, H = initialize(V, rank, seed=seed, norm=norm)
    iter = 0    
    while iter <= max_iter:
        targets = dl.get_targets(W.T,post)
        """
        #Add a function of counting #items in each cluster
        clusters = np.unique(targets)
        count_arr = [0 for i in range(0,len(clusters))]
        for c in targets:
            count_arr[c]+=1
        print sorted(count_arr)
        """
        if gt!=None:
            A = metrics.accuracy(gt,targets)
            F1 = metrics.f_measure(gt,targets)
            #print "Iter = %d, Acc = %f, F1 = %f" %(iter,A,F1)
        
        W, H = euclidean_update(V, W, H, norm)
        W, H = _adjustment(W, H)
        iter += 1

    return W, H

def euclidean_update(V, W, H, norm):
    """Update basis and mixture matrix based on Euclidean distance multiplicative update rules."""
    
    up = dot(W.T, V)
    down = dot(dot(W.T, W), H)
    elop_div = elop(up, down, div)
    
    H = multiply(H, elop_div)
    W = multiply(W, elop(dot(V, H.T), dot(W, dot(H, H.T)), div))
    
    return W, H

def _adjustment(W, H):
    """Adjust small values to factors to avoid numerical underflow."""
    H = max(H, np.finfo(H.dtype).eps)
    W = max(W, np.finfo(W.dtype).eps)
        
    return W, H

def initialize(V, rank, seed='random', norm='l2'):
    W,H = initialize_random(V, rank)
    if seed == "k-means":
        kmeans_results = kmeans(V.astype('float64'), rank, norm)
        labels = kmeans_results[0]
        H = kmeans_results[1]
        W = labels_to_matrix(labels, W)
    return W,H

# clustering using SVD
def svd(data, k, norm, post="k-means", latent_K = -1):
    data_norm = dl.norm_data(data, norm)
    
    if latent_K == -1:
        U, S, V_T = svds(data_norm, k, which = 'LM')
    else:
        U, S, V_T = svds(data_norm, latent_K, which = 'LM')
        
    U = sp.csr_matrix(U)
    targets = dl.get_targets(U.T,post,k=k)
    return targets