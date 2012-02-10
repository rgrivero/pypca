"""
    Module for Principal Component Analysis.
    
    Features:
    * pca and kernel pca
    * pca through singular value decomposition (SVD)

    Author: Alexis Mignon (c)
    Date: 10/01/2012
    e-mail: alexis.mignon@gmail.com
"""

import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigen_symmetric
from scipy.linalg import eigh

def full_pca(data):
    """
        Performs the complete eigen decomposition of
        the covariance matrix.
        
        arguments:
        * data: 2D numpy array where each row is a sample and
            each column a feature.
        
        return:
        * w: the eigen values of the covariance matrix sorted in from 
              highest to lowest.
        * u: the corresponding eigen vectors. u[:,i] is the vector
             corresponding to w[i]
             
        Notes: If you want to compute only a few number of principal
               components, you should consider using 'pca' or 'svd_pca'.
    """
    cov = np.cov(data.T)
    w,u = eigh(cov,overwrite_a = True)
    return w[::-1],u[:,::-1]

def pca(data,k):
    """
        Performs the eigen decomposition of the covariance matrix.
        
        arguments:
        * data: 2D numpy array where each row is a sample and
                each column a feature.
        * k: number of principal components to keep.
        
        return:
        * w: the eigen values of the covariance matrix sorted in from 
              highest to lowest.
        * u: the corresponding eigen vectors. u[:,i] is the vector
             corresponding to w[i]
             
        Notes: If the number of samples is much smaller than the number
               of features, you should consider the use of 'svd_pca'.
    """
    cov = np.cov(data.T)
    w,u = eigen_symmetric(cov,k = k,which = 'LA')
    return w[::-1],u[:,::-1]

def extern_pca(data,k):
    """
        Performs the eigen decomposition of the covariance matrix based
        on the eigen decomposition of the exterior product matrix.
        
        
        arguments:
        * data: 2D numpy array where each row is a sample and
                each column a feature.
        * k: number of principal components to keep.
        
        return:
        * w: the eigen values of the covariance matrix sorted in from 
              highest to lowest.
        * u: the corresponding eigen vectors. u[:,i] is the vector
             corresponding to w[i]
             
        Notes: This function computes PCA, based on the exterior product
               matrix (C = X*X.T/(n-1)) instead of the covariance matrix
               (C = X.T*X) and uses relations based of the singular
               value decomposition to compute the corresponding the
               final eigen vectors. While this can be much faster when 
               the number of samples is much smaller than the number
               of features, it can lead to loss of precisions.
               
               The (centered) data matrix X can be decomposed as:
                    X.T = U * S * v.T
               On computes the eigen decomposition of :
                    X * X.T = v*S^2*v.T
               and the eigen vectors of the covariance matrix are
               computed as :
                    U = X.T * v * S^(-1)
    """
    data_m = data - data.mean(0)
    K = np.dot(data_m,data_m.T)/(len(data)-1)
    w,v = eigen_symmetric(K,k = k,which = 'LA')
    U = np.dot(data.T,v/np.sqrt(w))
    return w[::-1],U[:,::-1]

def full_kpca(data):
    """
        Performs the complete eigen decomposition of a kernel matrix.
        
        arguments:
        * data: 2D numpy array representing the symmetric kernel matrix.
        
        return:
        * w: the eigen values of the covariance matrix sorted in from 
              highest to lowest.
        * u: the corresponding eigen vectors. u[:,i] is the vector
             corresponding to w[i]
             
        Notes: If you want to compute only a few number of principal
               components, you should consider using 'kpca'.
    """
    w,u = eigh(data,overwrite_a = True)
    return w[::-1],u[:,::-1]


def kpca(data,k):
    """
        Performs the eigen decomposition of the kernel matrix.
        
        arguments:
        * data: 2D numpy array representing the symmetric kernel matrix.
        * k: number of principal components to keep.
        
        return:
        * w: the eigen values of the covariance matrix sorted in from 
              highest to lowest.
        * u: the corresponding eigen vectors. u[:,i] is the vector
             corresponding to w[i]
             
        Notes: If you want to perform the full decomposition, consider 
               using 'full_kpca' instead.
    """
    w,u = eigen_symmetric(data,k = k,which = 'LA')
    return w[::-1],u[:,::-1]

class PCA(object):
    """
        PCA object to perform Principal Component Analysis.
    """
    def __init__(self,k = None, kernel = False,extern = False):
        """
            Constructor.
            
            arguments:
            * k: number of principal components to compute. 'None'
                 (default) means that all components are computed.
            * kernel: perform PCA on kernel matrices (default is False)
            * extern: use extern product to perform PCA (default is 
                   False). Use this option when the number of samples
                   is much smaller than the number of features.
        """
    
        self.k = k
        self.kernel = kernel
        self.extern = extern
    
    def fit(self,X):
        """
            Performs PCA on the data array X.
            arguments:
            * X: 2D numpy array. In case the array represents a kernel
                 matrix, X should be symmetric. Otherwise each row
                 represents a sample and each column represents a
                 feature.
        """
        if self.k is None :
            if self.kernel :
                pca_func = full_kpca
            else :
                pca_func = full_pca
            self.eigen_values_,self.eigen_vectors_ = pca_func(X)
        else :
            if self.kernel :
                pca_func = kpca
            elif self.extern :
                pca_func = svd_pca
            else :
                pca_func = pca
            self.eigen_values_,self.eigen_vectors_ = pca_func(X,k)
        
    if transform(self,X,whiten = False):
        """
            Project data on the principal components. If the whitening
            option is used, components will be normalized to that they
            have the same contribution.
            
            arguments:
            * X: 2D numpy array of data to project.
            * whiten: (default is False) all components are normalized
                so that they have the same contribution.
                
            returns:
            * prX : projection of X on the principal components.
            
            Notes: In the case of Kernel PCA, X[i] represents the value
               of the kernel between sample i and the j-th sample used
               at train time. Thus, if fit was called with a NxN kernel
               matrix, X should be a MxN matrix.
        """
        
        pr = np.dot(X,self.eigen_vectors_)
        if whiten :
            if self.kernel :
                pr /= self.eigen_values_
            else :
                pr /= np.sqrt(self.eigen_values_)
        return pr
