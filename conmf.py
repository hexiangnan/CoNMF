'''
Created on May 6, 2013
Last Update on Oct 9, 2014

CoNMF implementation for multi-view clustering proposed in paper [1].

If you use the codes, please cite the paper:
[1] Xiangnan He, Min-Yen Kan, Peichu Xie, and Xiao Chen. 2014. Comment-based multi-view clustering of web 2.0 items. 
In Proceedings of the 23rd international conference on World wide web (WWW '14).

@author: HeXiangnan (xiangnan@comp.nus.edu.sg)
'''
import scipy as sp
import numpy as np
import scipy.sparse as sparse
import data_load as dl
import metrics
from operator import div
from utils import dot,multiply,elop,initialize_random,labels_to_matrix
from kmeans import kmeans

np.set_printoptions(threshold='nan')
sp.set_printoptions(threshold='nan')

def conmf(datas, cluster_size, weights, regu_weights, norm="l2", seed="k-means", post="direct", method="pair-wise", gt = None): 
    """
    CoNMF on multi-view dataset (represented as datas).
    
    :param datas (type: list<csr_matrix>). [data_k] 
        Each element data_k is a sparse matrix: `scipy.sparse` of format csr
    :param cluster_size (type: int). Number of clusters 
    :param weights (type: list<int>). [weight_s]. 
        Each element weight_s is an integer, denoting the weight of the view in CoNMF factorization (i.e. \lambda_s in paper [1])
    :param regular_weights (type: list, 2-dimension). [[weight_st]]
        Each element weight_st is an integer, denoting the weight of view pair <s,t> in CoNMF regularization (i.e. \lambda_st in paper [1])
    :param norm (type: string). Normalization scheme in CoNMF initialization and factorization. Values can be 'l2', 'l1' and 'l0':
        'l2': each item vector is normalized by its Euclidean length (i.e. l2 distance).
        'l1': each item vector is normalized by its sum of all elements (i.e. l1 distance). 
        'l0': the whole matrix is normalized by the sum of all elements (i.e. l1 normalization on the whole matrix).
    :param seed (type: string). Initialization method in CoNMF. Values can be 'k-means' and 'random':
        'k-means': initialize W and H matrix using k-means results. The details are seen in paper [1] Section 4.5
        'random': randomly initialize W and H matrix. 
    :param post (type: string). Post processing on W matrix (m*k) to generate clustering result. Values can be 'direct' and 'k-means':
        'direct': for each item vector (m*1), use the element with largest value as its cluster assignment.
        'k-means': perform k-means on W matrix to get cluster assignment. 
    :param method (type: string). Regularization method of CoNMF. Currently support two ways:
        'pair-wise': pair-wise NMF, details in paper [1] Section 4.3
        'cluster-wise': cluster-wise NMF, details in paper [1] Section 4.4
        Note: experiments in paper [1] indicates 'pair-wise' performs slightly better than 'cluster-wise'. And also, 'pair-wise' is more efficient. 
    :param gt (type: string). Groundtruth of clustering (type: list). Each element gt_i represents the cluster assignment of item_i. 
        Note: this variable is only used for monitoring the performance in each iteration of CoNMF. 
        If the value is none, remember to comment the codes that use the variable, otherwise the program may crash.
        
    Return:
    targets (type: list<int>): Cluster assignment of items. Each element represents the item's cluster id. 
    Ws (type: list<csr_matrix>): W matrix of each view after CoNMF. Each element is a sparse matrix denoting the view's W matrix. 
    Hs (type: list<csr_matrix>): H matrix of each view after CoNMF. Each element is a sparse matrix denoting the view's H matrix.
    """
    
    if(len(datas)!=len(weights)):
        print "Error! Length of datas != length of weights."
        return None
    
    # Normalize the data of each view.
    datas_norm = []
    for i in range(0,len(datas)):
        data_norm = dl.norm_data(datas[i],norm)
        datas_norm.append(data_norm)
    
    Ws,Hs = conmf_factorize(method, datas_norm, weights, regu_weights, seed, post, norm, 100, cluster_size, gt)
    
    # By default, use the clustering result in last view as output for eval.
    targets = dl.get_targets(Ws[-1].T, post)
    return targets,Ws,Hs

def conmf_factorize(method, datas, weights, regu_weights, seed, post, norm, max_iter, rank, gt=None):
    """
    Factorization process of CoNMF.
    
    :param max_iter (type: int). Maximum iterations of executing CoNMF update rules.
    :param rank (type: int). Number of latent factors in NMF factorization. For clustering application, it is typicall set as the number of clusters.
    
    Other parameters are with same meaning of conmf().
    """
    if method not in ["pair-wise", "cluster-wise"]:
        print "Error! Method not in [pair-wise, cluster-wise]!"
        return None
    
    Ws, Hs = conmf_initialize(datas, rank, seed, weights, norm)

    targets, As, F1s = [],[],[]
    iter_num = 0
    while iter_num <= max_iter:
        targets = [dl.get_targets(W.T,post) for W in Ws]
        As = ["{0:.4f}".format(metrics.accuracy(gt, target)) for target in targets]
        F1s= ["{0:.4f}".format(metrics.f_measure(gt, target)) for target in targets]
        if iter_num==0:
            print "\t\t CoNMF Inits \t Acc = %s;\t F1 = %s " %(str(As), str(F1s))
        #print "\t\t Iter = %d: \t Acc = %s;\t F1 = %s " %(iter_num, str(As), str(F1s))
        Ws, Hs = conmf_update(datas, Ws, Hs, weights, regu_weights, norm, method)
        #cost = conmf_cost(Vs,Ws,Hs, weights, mutual_weights, norm, method)
        if iter_num==max_iter:
            print "\t\t CoNMF Ends \t Acc = %s;\t F1 = %s " %(str(As), str(F1s))
        iter_num += 1
    return Ws, Hs

def conmf_initialize(Vs, rank, seed, weights, norm):
    """
    Initialization for CoNMF, returning Ws, Hs.
    :param Vs (type: list<csr_matrix>). input views (i.e. parameter datas in conmf()) to run CoNMF for. 
    
    Other parameters are with same meaning of conmf().
    """
    #print "\t Initializing for CoNMF using %s" %seed
    length = len(Vs)
    W = initialize_random(Vs[-1], rank)[0]
    if seed == "k-means":
        labels = kmeans(dl.combineData(Vs, weights, norm) ,rank, norm)[0]
        #labels = kmeans(Vs[-1].astype('float64'), rank, norm)[0]
        W = labels_to_matrix(labels, W)
        Ws = [W for i in range(0,length)]
        Hs = [kmeans(Vs[i].astype('float64'), rank, norm)[1] for i in range(0,length)] # use the kmeans centroids on each single view as the init. of H
        return Ws, Hs
    if seed =="k-means|sepa":
        Ws = [labels_to_matrix(kmeans(Vs[i].astype('float64'), rank, norm)[0], W) for i in range(0,length)]
        Hs = [kmeans(Vs[i].astype('float64'), rank, norm)[1] for i in range(0,length)] # use the kmeans centroids on each single view as the init. of H
        return Ws, Hs

    Ws = [W for i in range(0,length)]
    Hs = [initialize_random(Vs[i],rank, )[1] for i in range(0,length)]
    return Ws, Hs

def conmf_cost(Vs, Ws, Hs, weights, regu_weights, norm, method):
    """
    Calculate the value of cost function(Frobenius form) of CoNMF.
    Two parts of the cost function: factorization part and regularization part, are calculated seperately.
    """
    # Factorization part
    sum1 = 0 
    for k in range(0, len(weights)):
        V, W, H = Vs[k],Ws[k],Hs[k]
        dot_WH = dot(W, H) 
        R = V - dot_WH
        sum1 = sum1 + weights[k] * multiply(R, R).sum()
    
    # Regularization part
    sum2 = 0 
    for k in range(0,len(weights)):
        for t in range(0,len(weights)):
            if k == t: continue
            if k<t: lambda_kt = regu_weights[k][t]
            if k>t: lambda_kt = regu_weights[t][k]
            if method == "pair-wise":
                R = Ws[k] - Ws[t]
                sum2 = sum2 + lambda_kt * multiply(R, R).sum()
            if method == "cluster-wise":
                R = dot(Ws[k].T, Ws[k]) - dot(Ws[t].T, Ws[t])
                sum2 = sum2 + lambda_kt * multiply(R, R).sum()
    
    return sum1 + sum2

def conmf_update(Vs, Ws, Hs, weights, regu_weights, norm, method):
    """
    The iterative rules of CoNMF. Details in paper [1] Section 4.3 and 4.4.
    
    :param Vs (type: list<csr_matrix>). Data of each view.
    :param Ws (type: list<csr_matrix). W matrix of each view.
    :param Hs (type: list<csr_matrix). H matrix of each view.
    :param weights (type: list<int>). Regularization parameters \lambda_s.
    :param regu_weights (type: list, 2 dimension). Regularization parameters \lambda_st.
    :param norm (type: string). Normalization scheme in CoNMF initialization and factorization. Values can be 'l2', 'l1' and 'l0':
        'l2': each item vector is normalized by its euclidean length (i.e. l2 distance).
        'l1': each item vector is normalized by its sum of all elements (i.e. l1 distance). 
        'l0': the whole matrix is normalized by the sum of all elements (i.e. l1 normalization on the whole matrix).
    :param method (type: string). Regularization method of CoNMF. Currently support two ways:
    'pair-wise': pair-wise NMF, details in paper [1] Section 4.3
    'cluster-wise': cluster-wise NMF, details in paper [1] Section 4.4
    
    V = W H
    V: #item * #feature; 
    W: #item * #cluster
    H: #cluster * #feature
    """
    # Update Hs[k]
    Ws, Hs = conmf_normalize(Ws, Hs, norm, basis="W")
    # Update Hs[k] first. NMF, CoNMF(mutual, cluster) are same for this updates
    for k in range(0,len(weights)):
        V, W, H = Vs[k],Ws[k],Hs[k]
        up = dot(W.T, V)
        down = dot(dot(W.T, W), H)
        elop_div = elop(up, down, div)
        Hs[k] = multiply(H, elop_div)

    # Update Ws[k]
    for k in range(0,len(weights)):
        V, W, H = Vs[k],Ws[k],Hs[k]
        up = dot(V, H.T) * weights[k]
        down = dot(W, dot(H, H.T)) * weights[k]
        if method == "pair-wise":
            for t in range(0,len(weights)):
                if k == t: continue
                if k<t: lambda_kt = regu_weights[k][t]
                if k>t: lambda_kt = regu_weights[t][k]
                up   = up   + Ws[t]*lambda_kt
                down = down + W*lambda_kt
        if method == "cluster-wise":
            for t in range(0,len(weights)):
                if k==t: continue
                if k<t: lambda_kt = regu_weights[k][t]
                if k>t: lambda_kt = regu_weights[t][k]
                up   = up   + 2*lambda_kt * dot(Ws[k], dot(Ws[t].T, Ws[t]))
                down = down + 2*lambda_kt * dot(Ws[k], dot(Ws[k].T, Ws[k]))
        elop_div = elop(up, down, div)
        Ws[k] = multiply(W, elop_div)

    return Ws, Hs
    
def conmf_normalize(Ws, Hs, norm="l2", basis="H"):
    """
    Normalization strategy of CoNMF, details in [1] Section 4.3.2.     
    
    :param basis (type: string). Indicate which matrix to use for calculating the norm factors. Values can be 'H' and 'W':
        'H':First calculating norm factors from H (rows), then
            1. Normalize W;
            2. Normalize H (row)
        'W':First calculating norm factors from W (cols), then
            1. Normalize H;
            2. Normalize W (col)
    """
    if norm == "none":
        return Ws, Hs
    
    if basis not in ["W","H"]:
        print "Error! Input basis is not 'W' or 'H'!"        
    
    if basis == "H":
        for k in range(len(Hs)):
            W, H = Ws[k],Hs[k]
            if norm == "l1" or norm =="l0":
                S = np.squeeze(np.asarray(H.sum(axis=1)))#Type: np.array
            if norm == "l2":
                S = np.squeeze(np.asarray(multiply(H,H).sum(axis=1)))
                S = np.sqrt(S)
                
            D,D_inv = sparse.lil_matrix((len(S),len(S))),sparse.lil_matrix((len(S),len(S)))
            D.setdiag(S)
            D_inv.setdiag(1.0/S)
            Ws[k] = dot(W,D)
            Hs[k] = dot(D_inv,H)
            
    if basis == "W":
        for k in range(len(Hs)):
            W, H = Ws[k],Hs[k]
            if norm == "l1" or norm =="l0":
                S = np.squeeze(np.asarray(W.sum(axis=0)))#Type: np.array
            if norm == "l2":
                S = np.squeeze(np.asarray(multiply(W,W).sum(axis=0)))
                S = np.sqrt(S)

            D,D_inv = sparse.lil_matrix((len(S),len(S))),sparse.lil_matrix((len(S),len(S)))
            D.setdiag(S)
            D_inv.setdiag(1.0/S)
            Hs[k] = dot(D,H)
            Ws[k] = dot(W,D_inv)
            
    return Ws,Hs
    