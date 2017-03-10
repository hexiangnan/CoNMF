'''
Created on May 6, 2013
Last Update on Oct 9, 2014

Wrap k-means (implemented by sklearn) clustering algorithm .

@author: HeXiangnan (xiangnan@comp.nus.edu.sg)
'''

from sklearn.cluster import KMeans
import data_load as dl
from scipy.sparse import csr_matrix

def kmeans(data, k, norm="l2", n_init = 1):
    """
    data: matrix, #item * #feature
    """
    if norm == None:
        km_model = KMeans(n_clusters=k, init='random', max_iter=500, n_init = n_init, n_jobs = 1, verbose = False)
        km_model.fit(data)
        return km_model.labels_, km_model.cluster_centers_
        
    data_norm = dl.norm_data(data, norm)
    km_model = KMeans(n_clusters=k, init='random', max_iter=500, n_init = n_init, n_jobs = 1, verbose = False)
    km_model.fit(data_norm)
    # km_model.cluster_centers_ is k*N of <type 'numpy.ndarray'>
    # H: converted km_model.cluster_centers_ to csr_matrix, shape: k*N
    H = csr_matrix(km_model.cluster_centers_)
    H = H.todense()
    H = H + 0.1 # Add a small number to each element of the centroid matrix
    H_norm = dl.norm_data(H, norm)
    
    return km_model.labels_, H_norm