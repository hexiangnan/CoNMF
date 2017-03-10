"""
Created on May 6, 2013
Last Update on Oct 9, 2014

Utilities to evaluate the clustering performance.
Self implemented evaluation metrics:
    accuracy;
    f_measure;
    purity;
    random_index;
Codes adopted from sklearn include:
    normalized_mutual_info_score();
    adjusted_rand_score();

For all metric functions, the inputs are:
    :param: labels_true (type list): the groundtruth (GT) of cluster assignment. Each element denotes an item's GT cluster_id. 
    :param: labels_pred (type list): the predicted cluster assignments. Each element denotes an item's predicted cluster_id.
    
@author: HeXiangnan (xiangnan@comp.nus.edu.sg)
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.misc import comb
from math import log
from munkres import Munkres # https://pypi.python.org/pypi/munkres/

def accuracy(labels_true, labels_pred):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0
            or classes.shape[0] == clusters.shape[0] == len(labels_true)):
        return 1.0
    
    # print "accuracy testing..."
    contingency = contingency_matrix(labels_true, labels_pred) #Type: <type 'numpy.ndarray'>:rows are clusters, cols are classes
    contingency = -contingency
    #print contingency
    contingency = contingency.tolist()
    m = Munkres() # Best mapping by using Kuhn-Munkres algorithm
    map_pairs = m.compute(contingency) #best match to find the minimum cost
    sum_value = 0
    for key,value in map_pairs:
        sum_value = sum_value + contingency[key][value]
    
    return float(-sum_value)/n_samples

def f_measure(labels_true, labels_pred): #Return the F1 score
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0
            or classes.shape[0] == clusters.shape[0] == len(labels_true)):
        return 1.0

    contingency = contingency_matrix(labels_true, labels_pred)
    #print contingency
    # Compute the ARI using the contingency data
    TP_plus_FP = sum(comb2(n_c) for n_c in contingency.sum(axis=1)) # TP+FP
    
    TP_plus_FN = sum(comb2(n_k) for n_k in contingency.sum(axis=0)) #TP+FN
    
    TP = sum(comb2(n_ij) for n_ij in contingency.flatten()) #TP
    
    #print "TP = %d, TP_plus_FP = %d, TP_plus_FN = %d" %(TP,TP_plus_FP,TP_plus_FN)
    P = float(TP) / TP_plus_FP
    R = float(TP) / TP_plus_FN
    
    return 2*P*R/(P+R) 

def normalized_mutual_info_score(labels_true, labels_pred):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    # Special limit cases: no clustering since the data is not split.
    # This is a perfect match hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0):
        return 1.0
    contingency = contingency_matrix(labels_true, labels_pred)
    contingency = np.array(contingency, dtype='float')
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred,
                           contingency=contingency)
    # Calculate the expected value for the mutual information
    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)
    nmi = mi / max(np.sqrt(h_true * h_pred), 1e-10)
    return nmi

def purity(labels_true, labels_pred):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0
            or classes.shape[0] == clusters.shape[0] == len(labels_true)):
        return 1.0
    
    #print "purity testing..."
    contingency = contingency_matrix(labels_true, labels_pred) #Type: <type 'numpy.ndarray'>:rows are clusters, cols are classes
    #print contingency
    
    cluster_number = contingency.shape[0]
    sum_ = 0
    for k in range(0,cluster_number):
        row = contingency[k,:]
        max_ = np.max(row)
        sum_ += max_
    return float(sum_) / n_samples

def random_index(labels_true, labels_pred):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0
            or classes.shape[0] == clusters.shape[0] == len(labels_true)):
        return 1.0

    contingency = contingency_matrix(labels_true, labels_pred)
    #print contingency
    # Compute the ARI using the contingency data
    TP_plus_FP = sum(comb2(n_c) for n_c in contingency.sum(axis=1)) # TP+FP
    
    TP_plus_FN = sum(comb2(n_k) for n_k in contingency.sum(axis=0)) #TP+FN
    
    TP = sum(comb2(n_ij) for n_ij in contingency.flatten()) #TP
    FP = TP_plus_FP - TP
    FN = TP_plus_FN - TP
    sum_all = comb2(n_samples)
    TN = sum_all - TP - FP - FN
    #print "TN = "+str(TN)
    
    return float(TP+TN) / (sum_all)

def adjusted_rand_score(labels_true, labels_pred):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0
            or classes.shape[0] == clusters.shape[0] == len(labels_true)):
        return 1.0

    contingency = contingency_matrix(labels_true, labels_pred)
    #print contingency
    # Compute the ARI using the contingency data
    #print "contingency.sum(axis=1) = %s" %(str(contingency.sum(axis=1)))
    sum_comb_c = sum(comb2(n_c) for n_c in contingency.sum(axis=1)) # TP+FP
    #print "sum_comb_c = %d" %(sum_comb_c)
    
    #print "contingency.sum(axis=0) = %s" %(str(contingency.sum(axis=0)))
    sum_comb_k = sum(comb2(n_k) for n_k in contingency.sum(axis=0)) #TP+FN
    #print "sum_comb_k = %d" %(sum_comb_k)
    
    #print "contingency.flatten() = %s" %(str(contingency.flatten()))
    sum_comb = sum(comb2(n_ij) for n_ij in contingency.flatten()) #TP
    #print "sum_comb = %d" %(sum_comb)
    
    prod_comb = (sum_comb_c * sum_comb_k) / float(comb(n_samples, 2))
    mean_comb = (sum_comb_k + sum_comb_c) / 2.
    
    return ((sum_comb - prod_comb) / (mean_comb - prod_comb))

"""
The following functions are used by the evaluation metrics functions only.
"""

def check_clusterings(labels_true, labels_pred):
    """Check that the two clusterings matching 1D integer arrays"""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0]))
    return labels_true, labels_pred

def contingency_matrix(labels_true, labels_pred, eps=None):
    """Build a contengency matrix describing the relationship between labels.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate

    eps: None or float
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propogation.
        If ``None``, nothing is adjusted.

    Returns
    -------
    contingency: array, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
    """
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = coo_matrix((np.ones(class_idx.shape[0]),
                              (class_idx, cluster_idx)),
                             shape=(n_classes, n_clusters),
                             dtype=np.int).toarray()
    if eps is not None:
        # don't use += as contingency is integer
        contingency = contingency + eps
    return contingency

def mutual_info_score(labels_true, labels_pred, contingency=None):
    """Mutual Information between two clusterings

    See also
    --------
    adjusted_mutual_info_score: Adjusted against chance Mutual Information
    normalized_mutual_info_score: Normalized Mutual Information
    """
    if contingency is None:
        labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
        contingency = contingency_matrix(labels_true, labels_pred)
    contingency = np.array(contingency, dtype='float')
    contingency_sum = np.sum(contingency)
    pi = np.sum(contingency, axis=1)
    pj = np.sum(contingency, axis=0)
    outer = np.outer(pi, pj)
    nnz = contingency != 0.0
    # normalized contingency
    contingency_nm = contingency[nnz]
    log_contingency_nm = np.log(contingency_nm)
    contingency_nm /= contingency_sum
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    log_outer = -np.log(outer[nnz]) + log(pi.sum()) + log(pj.sum())
    mi = (contingency_nm * (log_contingency_nm - log(contingency_sum))
          + contingency_nm * log_outer)
    return mi.sum()

def entropy(labels):
    """Calculates the entropy for a labeling."""
    if len(labels) == 0:
        return 1.0
    label_idx = np.unique(labels, return_inverse=True)[1]
    pi = np.bincount(label_idx).astype(np.float)
    pi = pi[pi > 0]
    pi_sum = np.sum(pi)
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    return -np.sum((pi / pi_sum) * (np.log(pi) - log(pi_sum)))

def comb2(n):
    # the exact version is faster for k == 2: use it by default globally in
    # this module instead of the float approximate variant
    return comb(n, 2, exact=1)

def get_map_pairs(labels_true, labels_pred):
    """
        Given the groundtruth labels and predicted labels, get the best mapping pairs by Munkres algorithm.
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    #n_samples = labels_true.shape[0]
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0
            or classes.shape[0] == clusters.shape[0] == len(labels_true)):
        return 1.0
    
    # print "accuracy testing..."
    contingency = contingency_matrix(labels_true, labels_pred) #Type: <type 'numpy.ndarray'>:rows are clusters, cols are classes
    contingency = -contingency
    #print contingency
    contingency = contingency.tolist()
    m = Munkres() # Best mapping by using Kuhn-Munkres algorithm
    map_pairs = m.compute(contingency) #best match to find the minimum cost
    return map_pairs

if __name__ == '__main__':
    # Test evaluation metrics
    labels = np.array([1,1,1,1,1,1,3,3,3,3,3,3,2,2,2,2,2])
    gt = np.array([1,1,1,1,1,2,1,2,2,2,2,3,1,1,3,3,3])
    print accuracy(gt, labels) #0.7059
    print f_measure(gt,labels) #0.4762
    print normalized_mutual_info_score(gt, labels) #0.365
    print purity(gt,labels) #0.7059
    print random_index(gt,labels) #0.6765

    labels = np.array([2,2,1,1,1,1,1,2])
    gt = np.array([1,1,2,2,3,3,3,3])
    print accuracy(gt, labels) #0.625
    print f_measure(gt,labels) #0.4762
    print normalized_mutual_info_score(gt, labels) #0.4587
    print purity(gt,labels) #0.875
    print random_index(gt,labels) #0.6071
    
    print "End"