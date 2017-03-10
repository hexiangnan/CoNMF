'''
Created on May 8, 2013
Last Update on Oct 9, 2014

Utilities for loading data.

If you use the codes, please cite the paper:
[1] Xiangnan He, Min-Yen Kan, Peichu Xie, and Xiao Chen. 2014. Comment-based multi-view clustering of web 2.0 items. 
In Proceedings of the 23rd international conference on World wide web (WWW '14).

@author: HeXiangnan (xiangnan@comp.nus.edu.sg)
'''
import ast
from sklearn.utils import check_arrays
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from scipy.io import loadmat, savemat
from sklearn import metrics
from sklearn.cluster import KMeans
import random

def fit_transform(raw_data, weight_type = "count"):
    """
    Apply term weighting (e.g. tf, tf-idf) on raw_data. 
    Input:
        raw_data (type: list). [[features], [features]...
        weight_type (type: string). Supported methods include "count","tf","tf-idf", "tfidf", "tf-df","tfdf".
    Return:
        data: coo_matrix, #items * #features
        feature_names: [feature_name]
        feature_scores: np.array([feature_score])
    """
    if weight_type not in ["count","tf","tf-idf", "tfidf", "tf-df","tfdf"]:
        print "Error! Input weight_type is not in [count,tf,tf-idf, tf-df]!"
        return None
    N = len(raw_data) # N denotes #items
    # scan the raw_data, build the feature space
    dict_feature, set_feature, feature_names = {}, set(),[] # map<feature_name, id>
    for features in raw_data:
        set_feature |= set(features)
    M = len(set_feature)
    i = 0
    for feature in sorted(set_feature):
        dict_feature[feature] = i
        feature_names.append(feature)
        i = i+1
    # scan the raw_data, build the count matrix
    data_count = sp.dok_matrix((N,M),dtype='float64')
    df = np.array([0.0 for i in range(0,len(feature_names))]) # df score of each feature
    tf = np.array([0.0 for i in range(0,len(feature_names))]) # term frequency of each feature (no log imposed)
    for i in range(0,len(raw_data)):
        features = raw_data[i]
        for feature in features:
            feature_id = dict_feature[feature]
            data_count[i,feature_id] += 1.0
            tf[feature_id] += 1.0
        for feature in set(features):
            feature_id = dict_feature[feature]
            df[feature_id] += 1.0
    
    if weight_type == "count":
        return data_count.tocoo(),feature_names,df
    if weight_type == "tf":
        data_tf = data_count.tocoo()
        data_tf.data = 1 + np.log2(data_tf.data)
        tf = 1 + np.log2(tf)
        return data_tf, feature_names, tf
    if weight_type in ["tf-idf","tfidf"]:
        idf =  np.log2(N/df)# idf score of each feature, i.e. log(N/n_i)
        items = data_count.items()
        for (i,j), f_ij in items:
            data_count[i,j] = (1 + np.log2(f_ij)) * idf[j]
        tfidf = tf * idf
        return data_count,feature_names,tfidf
    if weight_type in ["tf-df","tfdf"]:
        items = data_count.items()
        for (i,j), f_ij in items:
            data_count[i,j] = (1 + np.log2(f_ij)) * df[j]
        tfdf = tf * df
        return data_count,feature_names,tfdf
    
def loadTargets(filename="lastfm/item_descwords_dict.txt", data_size = -1, items_subset = set()):
    #print "Loading targets of items from item_descwords_dict.txt..."
    fr = file(filename,"r")
    lines = fr.readlines()
    fr.close()
    target = []
    i = 0
    for line in lines:
        if len(items_subset)==0 or (i in items_subset):
            target.append(int(line.split("|")[1])-1)
            if(data_size > 0 and i>=data_size):
                break
        i = i+1
    target = np.array(target)
    target = target - min(target) # To ensure the targets are beginning from 0
    return target

def write_features(filename, feature_names):
    """
    Write <feature_name, id(from 0, corresponding to the columns of the generated matrix)
    """
    fw = file(filename, "w")
    for i in range(0,len(feature_names)):
        fw.write("%d\t%s\n" %(i, feature_names[i]))
    fw.close()
    print "Wrote <feature, id> to %s" %filename
    
def loadData(filename, weight_type = "count", feature_perc=1.0, data_size = -1, items_subset = set()):
    """
    filename: the name of data file
    weight_type: weight type of tokens: "count","tf-idf","tf"
    feature_pect: percentage of features to select according to its weighted values. (count: #document contains the feature; tfidf: tf-idf value)
    data_size: #data to select. default is -1, means select all data.
    item_subset: only read lines in the item_subset, which is a set of line numbers
    """
    weight_type = weight_type.lower()
    print "Loading data from %s, weight_type = %s" %(filename,weight_type)
    raw_data = []
    lines = file(filename, "r").readlines()
    i = 0
    for line in lines:
        if len(items_subset)==0 or (i in items_subset):
            line = line.replace('\n','')
            arr = ast.literal_eval(line.split("|")[2]) #converted to list
            raw_data.append(arr)
            if(data_size > 0 and i>=data_size): break
        i = i+1
    data, feature_names, features_score = fit_transform(raw_data, weight_type = weight_type) # type of data is <class 'scipy.sparse.coo.coo_matrix'>
    #write_features("%s_features" %filename, feature_names)
    # feature selection
    # Note: after selection, the vocabulary of features names are changed! Not modify them yet!
    if feature_perc<1:
        n_features = data.shape[1]
        upbound = n_features*feature_perc
        dict_feat_score ={}
        for i in range(0,len(features_score)):
            dict_feat_score[i] = features_score[i]
        sorted_list = sorted(dict_feat_score.items(), key=lambda d: d[1], reverse=True) #Sort the dict by its values, descending(reverse=True),ascending(reverse=False)
        selected_features=[]
        i = 0
        for key,value in sorted_list:
            selected_features.append(key)
            i = i+1
            if i>upbound:
                break
        data = data.tocsc()
        data = data[:,selected_features]
        print "\t Selected top %d(%f) from %d features:"%(len(selected_features),feature_perc,n_features)
    # print vectorizer.get_features_tfidf()
    # print vectorizer.vocabulary_
    data = check_arrays(data, sparse_format="csr", copy=False, dtype=np.float64)[0] # convert to the type: csr sparse matrix
    # Change: Normalization is done in each method
    # data_norms = normalize(data,'l2',axis=1,copy=False) # Squared euclidean norm(l2-norm) of each data point.
    return data#.astype('f') #convert type to float32 to save space

# weights can only be integers
def loadMultiData(filenames, weights, weight_types,isMergeFeatures=False, data_size = -1):
    print "Loading data from multiple files: weights=%s, types=%s" %(str(weights),str(weight_types))
    datas = []
    for index in range(0,len(filenames)):
        filename = filenames[index]
        weight_type = weight_types[index]
        data = loadData(filename,weight_type=weight_type,data_size=data_size)
        datas.append(data)
    # Combine multiple matrices
    # Does not merge features. Just concatenate all feature spaces
    if isMergeFeatures == False:
        combined = datas[0] * weights[0]
        for i in range(1,len(datas)):
            data = datas[i]
            data = data*weights[i]
            combined = sp.hstack([combined,data])
        combined = check_arrays(combined, sparse_format="csr", copy=False, dtype=np.float64)[0] # convert to the type: csr sparse matrix
        # Changed: normalization is done in each method
        # combined_norms = normalize(combined,'l2',axis=1,copy=False) # Squared euclidean norm of each data point.
        return combined#.astype('f') #convert type to float32 to save space
    else: #Donot implement the merging features version yet.
        pass

def combineData(datas, weights, norm):   
    """
    First normalize each view, then combine
    """
    combined = norm_data(datas[0],norm)*weights[0]
    for i in range(1,len(datas)):
        data = norm_data(datas[i],norm)*weights[i]
        combined = sp.hstack([combined,data])
    combined = check_arrays(combined, sparse_format="csr", copy=False, dtype=np.float64)[0] # convert to the type: csr sparse matrix
    # Changed: normalization is done in each method
    # combined_norms = normalize(combined,'l2',axis=1,copy=False) # Squared euclidean norm of each data point.
    return combined#.astype('f') #convert type to float32 to save space

def loadItems(filename, data_size = -1):
    fr = file(filename,"r")
    lines = fr.readlines()
    fr.close()
    items = []
    i = 1
    for line in lines:
        items.append(str(line.split("|")[0]))
        if(data_size>0 and i>data_size):
            break
        i = i+1
    
    return items

def get_targets(H, method="k-means", k = 21): # k is #cluster. H is the k*n coefficient matrix
    if method.lower() not in ["k-means","kmeans","direct"]:
        print "Error! The method of post: data_load.get_targets is wrong!"
    col = H.shape[1]
    if method == "k-means" or method=="kmeans":
        H = H.todense()
        H[np.isnan(H)]=0
        H = normalize(H.T,'l2',axis=1,copy=False)
        
        km_model = KMeans(n_clusters=k, init='random', max_iter=500, n_init = 10, n_jobs = 1, verbose = False)
        km_model.fit(H)
        return km_model.labels_
    if method == "direct":
        if sp.issparse(H)==True:
            H = H.todense()
        H = np.array(H)
        targets = []
        for i in range(0,col):
            h_i = H[:,i]
            col_i = h_i.tolist()
            cluster = col_i.index(max(col_i))
            targets.append(cluster)
        return np.array(targets)

def filterData(data, low_freq = -1):
    """
        Remove features with #items <= low_freq
        data: a matrix
    """
    if low_freq < 0:
        return data
    else:
        data_f = data.tocsc()
        selected_features = []
        item_number, feature_number = data_f.shape
        dict_feature_items={}
        # Build the dict_user_items, a map <user_id, set_of_items>
        for i in range(item_number):
            row = data_f.getrow(i)
            indices = row.indices
            for j in indices:
                if dict_feature_items.has_key(j) == False:
                    dict_feature_items[j] = set()
                dict_feature_items[j].add(i)
        # Select remained features
        for user, items in dict_feature_items.items():
            if len(items) > low_freq:
                selected_features.append(user)
        
        data_f = data_f[:,selected_features]
        return data_f
         
def loadMATdata(filename, view_names, data_size = -1):
    """
    Load data from .mat datasource.
    data1: #items * #features1
    data2: #items * #features1...
    gnd : #items*1
    
    return:
    [datas, gt, #cluster]
    """
    datas,gt,lnames,dims = [],[],[],[]
    
    dict_mat = loadmat(filename)
    for name in sorted(dict_mat.keys()):
        if name.lower() == "gnd":
            gt = np.array(dict_mat[name].T)[0].astype('int32')
            if data_size > 0:   gt = np.array(dict_mat['gnd'].T)[0][0:data_size].astype('int32')
    if len(gt) == 0:
        print "Error! The input data source does not contain gnd!"
        return None
    
    for view_name in view_names:
        if dict_mat.has_key(view_name) == False:
            print "Error! Input view_name=%s is not in the data source!" %(view_name)
        else:
            data = sp.csr_matrix(dict_mat[view_name],dtype='float64')            
            if data_size > 0:   data = data[0:data_size,:]
            datas.append(data)
            lnames.append(view_name)
            dims.append(data.shape)

    k_cluster = len(np.unique(gt))
    print "Load views %s from %s, dims = %s, k = %d" %(lnames,filename.split("/")[-1],str(dims),k_cluster)
    return datas,lnames,gt,k_cluster

def norm_data(data,norm):
    """
    norm = 'l1', 'l2' or 'l0' or 'l2+l0'...
        'l0': normalize the data matrix as the sum of all entries=1
    """
    data_norm = data
    norms = norm.split("+")# the sequence of norm
    for norm in norms: 
        if norm in ['l1','l2']:
            data_norm = normalize(data_norm,norm,axis=1,copy=True) # donot change the original input data
        if norm == "l0":
            _sum = data.sum()
            data_norm = data_norm/_sum

    return data_norm