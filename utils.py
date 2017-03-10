"""
Created on May 6, 2013
Last Update on Oct 9, 2014

Utils of NMF and CoNMF.

Codes adopted from nimfa include:
    dot();
    multiply();
    elop();
    _op_spmatrix();
    __op_spmatrix();
    _op_matrix();
    
@author: HeXiangnan (xiangnan@comp.nus.edu.sg)
"""

import numpy as np
import scipy.sparse as sp
import warnings

def initialize_random(V, rank):
    """
    Randomly initiate W and H matrix to run NMF for V
    
    :param V: Data matrix to run NMF for.
    :type V:  class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
    :rank:    number of latent factors in NMF estimation, i.e. the rank of W and H matrix.
    :type rank: int.
    """
    max_V = V.data.max()
    W = np.mat(np.random.RandomState().uniform(0, max_V, (V.shape[0], rank)))    
    H = np.mat(np.random.RandomState().uniform(0, max_V, (rank, V.shape[1])))
    
    W = sp.csr_matrix(W)
    H = sp.csr_matrix(H)
    
    return W, H

def labels_to_matrix(labels, W):
    """
    Change the input W matrix s.t. the largest element of each item vector is its cluster assignment (as in labels).
    
    :param labels: the cluster assignment of items.
    :type labels: list. 
    :param W: the item latent matrix (m*k)
    :type W: class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
    """
    for i in range(0,len(labels)):
        j = labels[i]
        row_i = W.getrow(i)
        if row_i.getnnz() > 0:
            max_index = row_i.indices[np.argmax(row_i.data)]
            max_value = np.max(row_i.data)
            #swap the value of W[i,j] and W[i,max_index]
            t = W[i,j]
            W[i,j] = max_value
            W[i,max_index] = t
        else:
            W[i,j] = 1.0
    return W

def dot(X, Y):
    """
    Compute dot product of matrices :param:`X` and :param:`Y`.
    
    :param X: First input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
    :param Y: Second input matrix. 
    :type Y: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix` 
    """
    if sp.isspmatrix(X) and sp.isspmatrix(Y):
        return X * Y
    elif sp.isspmatrix(X) or sp.isspmatrix(Y):
        # avoid dense dot product with mixed factors
        return sp.csr_matrix(X) * sp.csr_matrix(Y)
    else:
        return np.asmatrix(X) * np.asmatrix(Y)

def multiply(X, Y):
    """
    Compute element-wise multiplication of matrices :param:`X` and :param:`Y`.
    
    :param X: First input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
    :param Y: Second input matrix. 
    :type Y: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix` 
    """
    if sp.isspmatrix(X) and sp.isspmatrix(Y):
        return X.multiply(Y)
    elif sp.isspmatrix(X) or sp.isspmatrix(Y):
        return _op_spmatrix(X, Y, np.multiply) 
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.multiply(np.mat(X), np.mat(Y))

def elop(X, Y, op):
    """
    Compute element-wise operation of matrix :param:`X` and matrix :param:`Y`.
    
    :param X: First input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
    :param Y: Second input matrix.
    :type Y: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
    :param op: Operation to be performed. 
    :type op: `func` 
    """
    try:
        zp1 = op(0, 1) if sp.isspmatrix(X) else op(1, 0)
        zp2 = op(0, 0) 
        zp = zp1 != 0 or zp2 != 0
    except:
        zp = 0
    if sp.isspmatrix(X) or sp.isspmatrix(Y):
        return _op_spmatrix(X, Y, op) if not zp else _op_matrix(X, Y, op)
    else:
        try:
            X[X == 0] = np.finfo(X.dtype).eps
            Y[Y == 0] = np.finfo(Y.dtype).eps
        except ValueError:
            return op(np.mat(X), np.mat(Y))
        return op(np.mat(X), np.mat(Y))
    
def _op_spmatrix(X, Y, op):
    """
    Compute sparse element-wise operation for operations preserving zeros.
    
    :param X: First input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
    :param Y: Second input matrix.
    :type Y: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
    :param op: Operation to be performed. 
    :type op: `func` 
    """
    # distinction as op is not necessarily commutative
    return __op_spmatrix(X, Y, op) if sp.isspmatrix(X) else __op_spmatrix(Y, X, op)

def __op_spmatrix(X, Y, op):
    """
    Compute sparse element-wise operation for operations preserving zeros.
    
    :param X: First input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia
    :param Y: Second input matrix.
    :type Y: :class:`numpy.matrix`
    :param op: Operation to be performed. 
    :type op: `func` 
    """
    assert X.shape == Y.shape, "Matrices are not aligned."
    eps = np.finfo(Y.dtype).eps if not 'int' in str(Y.dtype) else 0
    Xx = X.tocsr()
    r, c = Xx.nonzero()
    R = op(Xx[r,c], Y[r,c]+eps)
    R = np.array(R)
    assert 1 in R.shape, "Data matrix in sparse should be rank-1."
    R = R[0, :] if R.shape[0] == 1 else R[:, 0]
    return sp.csr_matrix((R, Xx.indices, Xx.indptr), Xx.shape)

def _op_matrix(X, Y, op):
    """
    Compute sparse element-wise operation for operations not preserving zeros.
    
    :param X: First input matrix.
    :type X: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
    :param Y: Second input matrix.
    :type Y: :class:`scipy.sparse` of format csr, csc, coo, bsr, dok, lil, dia or :class:`numpy.matrix`
    :param op: Operation to be performed. 
    :type op: `func` 
    """
    # operation is not necessarily commutative 
    assert X.shape == Y.shape, "Matrices are not aligned."
    eps = np.finfo(Y.dtype).eps if not 'int' in str(Y.dtype) else 0
    return np.mat([[op(X[i,j], Y[i,j] + eps) for j in xrange(X.shape[1])] for i in xrange(X.shape[0])])