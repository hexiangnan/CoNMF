'''
Created on May 6, 2013
Last Update on Oct 9, 2014

The main function for multi-view clustering evaluation in paper [1].

If you use the codes, please cite the paper:
[1] Xiangnan He, Min-Yen Kan, Peichu Xie, and Xiao Chen. 2014. Comment-based multi-view clustering of web 2.0 items. 
In Proceedings of the 23rd international conference on World wide web (WWW '14).

@author: HeXiangnan (xiangnan@comp.nus.edu.sg)
'''
import data_load as dl
import metrics
import numpy as np
import scipy.sparse as sp
from kmeans import kmeans
from time import time
from nmf import nmf,svd
from conmf import conmf
from scipy.io import loadmat, savemat
#import ast

n_runs = 20 # number of runs with random initialization for clustering evaluation.

def evaluation_scores(groundtruth, labels_pred):
    """
    Eval scores of the predicted results.
     
    :param: groundtruth (type list): the groundtruth (GT) of cluster assignment. Each element denotes an item's GT cluster_id. 
    :param: labels_pred (type list): the predicted cluster assignments. Each element denotes an item's predicted cluster_id.
    """
    NMI = metrics.normalized_mutual_info_score(groundtruth,labels_pred)
    A = metrics.accuracy(groundtruth,labels_pred)
    F1 = metrics.f_measure(groundtruth,labels_pred)
    P = metrics.purity(groundtruth,labels_pred)
    RI = metrics.random_index(groundtruth,labels_pred)
    ARI = metrics.adjusted_rand_score(groundtruth,labels_pred)
    map_pairs = metrics.get_map_pairs(groundtruth,labels_pred)
    return NMI, A, F1, P, RI, ARI, map_pairs
    
def single_evaluation(method, filename, view_names, norm = 'l2',seed="random", post = "direct", data_size=-1):
    """
    Eval of baselines on each single view.
    Implemented baselines include k-means, NMF and SVD.
    
    :param: method (type: string). Supported methods include "k-means", "nmf", "svd"
    :param: filename (type: string). The path of the input multi-view data (.MAT format).
    :param: view_names (type: list<string>). View names of the input multi-view data.
    :param: norm (type: string). Normalization strategy on the input data. Supported norm methods include:
        'l2': each item vector is normalized by its Euclidean length (i.e. l2 distance).
        'l1': each item vector is normalized by its sum of all elements (i.e. l1 distance). 
        'l0': the whole matrix is normalized by the sum of all elements (i.e. l1 normalization on the whole matrix).
    :param seed (type: string). The initialization method in CoNMF. Values can be 'k-means' and 'random':
        'k-means': initialize W and H matrix using k-means results. The details are seen in paper [1] Section 4.5
        'random': randomly initialize W and H matrix. 
    :param post (type: string). Post processing on W matrix (m*k) to generate clustering result. Values can be 'direct' and 'k-means':
        'direct': for each item vector (m*1), use the element with largest value as its cluster assignment.
        'k-means': perform k-means on W matrix to get cluster assignment. 
    :param data_size (type: int). Select the first data_size items to run clustering algorithm. 
        When the value is -1, the clustering algorithm is run on all items. 
        This parameter is for a quick check of clustering algorithms in case the input data is too large. 
    """
    method = method.lower()
    if method not in ["k-means","kmeans","nmf","svd"]:
        print "Error! Wrong input method name."
        return
    
    datas, names, groundtruth, cluster_k = dl.loadMATdata(filename, view_names,data_size)
    for i in range(0,len(datas)):
        data,view_name = datas[i],names[i]
        NMIs,As,F1s = [],[],[]
        print "Running single %s(k=%d,norm=%s,seed=%s,post=%s) for %s, size = %s, #runs: %d" %(method,cluster_k,norm,seed,post,view_name,str(data.shape), n_runs)
        i_run = 1
        t0 = time()
        while i_run <= n_runs:
            t1 = time()
            if method == "k-means" or method =="kmeans":
                labels = kmeans(data, cluster_k, norm)[0]
            if method == "nmf":
                labels = nmf(data, cluster_k, norm, seed, post, groundtruth)[0]
            if method == "svd":
                labels = svd(data, cluster_k, norm, post="k-means")
                
            NMI,A,F1,P,RI,ARI,map_pairs = evaluation_scores(groundtruth,labels)
            print "\t %d-th run(time=%ds),<Acc, F1, NMI>\t%f\t%f\t%f" %(i_run,time()-t1,A,F1,NMI)
            i_run = i_run+1
            NMIs.append(NMI)
            As.append(A)
            F1s.append(F1)
        
        print "Results of %d runs (mean,std_var):\n\t Acc: %f, %f\n\t F1 : %f, %f\n\t NMI: %f, %f"  %(n_runs,
            np.mean(As),np.std(As),np.mean(F1s),np.std(F1s),np.mean(NMIs),np.std(NMIs))
        print "Running time: %fs" %(time() - t0)  

def multi_evaluation(method, filename, view_names, weights, norm = 'l2', seed="random", post = "direct", data_size=-1):
    """
    Eval of baselines (k-means, nmf, svd) on the combined view of all views for multi-view clustering. 
    
    :param: method (type: string). Supported methods include "k-means", "nmf", "svd"
    :param: filename (type: string). The path of the input multi-view data (.MAT format).
    :param: view_names (type: list<string>). View names of the input multi-view data.
    :param: weights (type: list<int>). The weight of each view to build the combined view.
    :param: norm (type: string). Normalization strategy on the input data. Supported norm methods include:
        'l2': each item vector is normalized by its Euclidean length (i.e. l2 distance).
        'l1': each item vector is normalized by its sum of all elements (i.e. l1 distance). 
        'l0': the whole matrix is normalized by the sum of all elements (i.e. l1 normalization on the whole matrix).
    :param seed (type: string). The initialization method in CoNMF. Values can be 'k-means' and 'random':
        'k-means': initialize W and H matrix using k-means results. The details are seen in paper [1] Section 4.5
        'random': randomly initialize W and H matrix. 
    :param post (type: string). Post processing on W matrix (m*k) to generate clustering result. Values can be 'direct' and 'k-means':
        'direct': for each item vector (m*1), use the element with largest value as its cluster assignment.
        'k-means': perform k-means on W matrix to get cluster assignment. 
    :param data_size (type: int). Select the first data_size items to run clustering algorithm. 
        When the value is -1, the clustering algorithm is run on all items. 
        This parameter is for a quick check of clustering algorithms in case the input data is too large.      
    """
    method = method.lower()
    if method not in ["k-means","kmeans","nmf","svd"]:
        print "Error! Wrong input method name."
        return None
    # weights can only be integers
    if len(view_names)!= len(weights):
        print "Error! Length of view_names is not equal to the length of weights!"
        return None

    datas, names, groundtruth, cluster_k = dl.loadMATdata(filename, view_names,data_size)
    data = dl.combineData(datas, weights, norm)
    NMIs,As,F1s = [],[],[]
    print "Running multi- %s(k=%d,norm=%s,seed=%s,post=%s) for %s, size = %s, #runs: %d" %(method,cluster_k,norm,seed,post,names,str(data.shape), n_runs)
    print "view_names = %s, weights = %s" %(str(names),str(weights))
    i_run = 1
    t0 = time()
    while i_run <= n_runs:
        t1 = time()
        if method == "kmeans" or method=="k-means":
            labels = kmeans(data, cluster_k, norm)[0]
        if method == "nmf":
            labels = nmf(data, cluster_k, norm, seed, post, groundtruth)[0]
        if method == "svd":
            labels = svd(data, cluster_k, norm, post)
        NMI,A,F1,P,RI,ARI = evaluation_scores(groundtruth,labels)
        print "\t %d-th run(time=%ds),<Acc, F1, NMI>\t%f\t%f\t%f" %(i_run,time()-t1,A,F1,NMI)
        NMIs.append(NMI)
        As.append(A)
        F1s.append(F1)
        i_run = i_run+1    
            
    print "Results of %d runs (mean,std_var):\n\t Acc: %f, %f\n\t F1 : %f, %f\n\t NMI: %f, %f"  %(n_runs,
        np.mean(As),np.std(As),np.mean(F1s),np.std(F1s),np.mean(NMIs),np.std(NMIs))
    print "Running time: %fs" %(time() - t0)
    
# CoNMF evaluation
def conmf_evaluation(regu_method, filename, view_names, weights, regu_weights, norm = 'l2', seed = "random", post = "direct", data_size = -1):
    """
    Eval of CoNMF for multi-view clustering.
    
    :param: regu_method (type: string). Supported methods include "pair-wise", "cluster-wise"
    :param: filename (type: string). The path of the input multi-view data (.MAT format).
    :param: view_names (type: list<string>). View names of the input multi-view data.    
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
    :param data_size (type: int). Select the first data_size items to run clustering algorithm. 
        When the value is -1, the clustering algorithm is run on all items. 
        This parameter is for a quick check of clustering algorithms in case the input data is too large.      
    """        
    datas, names, groundtruth, cluster_k = dl.loadMATdata(filename, view_names, data_size)
        
    print "Running CoNMF-%s (k=%d,norm=%s,seed=%s,post=%s,weights=%s,regularization_weights=%s)"%(regu_method,cluster_k,norm,seed,post, str(weights),str(regu_weights))
    i_run = 1
    t0 = time()
    NMIs,As,F1s = [],[],[]
    while i_run<= n_runs:
        print "\t %d-th run starts. Initializing using %s ..." %(i_run, seed)
        t1 = time()
        results = conmf(datas, cluster_k, weights, regu_weights, norm, seed, post,regu_method, groundtruth)
        labels,Ws,Hs = results[0],results[1],results[2]
        NMI,A,F1,P,RI,ARI,map_pairs = evaluation_scores(groundtruth,labels)
        print "\t %d-th run ends (time=%ds). <Acc, F1, NMI>(last view)\t%f\t%f\t%f" %(i_run,time()-t1,A,F1,NMI)
        NMIs.append(NMI)
        As.append(A)
        F1s.append(F1)
        i_run = i_run+1    
        
    print "Results (on the last view) of %d runs (mean,std_var):\n\t Acc: %f, %f\n\t F1 : %f, %f\n\t NMI: %f, %f"  %(n_runs,
        np.mean(As),np.std(As),np.mean(F1s),np.std(F1s),np.mean(NMIs),np.std(NMIs))
    print "Running time: %fs" %(time() - t0)    

if __name__ == '__main__':

    weights = [1,1,1]
    regular_w = 1
    W = [[regular_w for x in range(len(weights))] for y in range(len(weights))] #regu_weights
        
    filename = "dataset/lastfm_tfidf.mat"
    view_names = ['descwords','commwords','users'] # Only the results on the last view are shown, so put the best view in the last.
    conmf_evaluation("pair-wise",filename,view_names,weights,W,norm="l2",seed="k-means",post="direct")
    #conmf_evaluation("cluster-wise",filename,view_names,weights,W,norm="l2",seed="k-means",post="direct")

    filename = "dataset/yelp_tfidf.mat"
    view_names = ['descwords','users','commwords'] # Only the results on the last view are shown, so put the best view in the last.
    conmf_evaluation("pair-wise",filename,view_names,weights,W,norm="l2",seed="k-means",post="direct")
    #conmf_evaluation("cluster-wise",filename,view_names,weights,W,norm="l2",seed="k-means",post="direct")

    """
    # Eval for baselines. 
    single_evaluation("k-means",filename,view_names,norm='l2',seed='random',post='direct')
    single_evaluation("nmf",filename,view_names,norm='l2',seed='random',post='direct')
    single_evaluation("svd",filename,view_names,norm='l2',seed='random',post='kmeans')
    
    multi_evaluation("k-means",filename,view_names, weights, norm='l2',seed='random',post='direct')
    multi_evaluation("nmf",filename,view_names, weights, norm='l2',seed='random',post='direct')
    multi_evaluation("svd",filename,view_names, weights, norm='l2',seed='random',post='kmeans')
    """
    
    print "End."