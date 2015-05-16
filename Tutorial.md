# Tutorial for the Principal Component Analysis python module.

# Introduction #

pypca is a simple Python module to perform Principal Component Analysis (PCA).

## Derivation ##

PCA consists in finding the "principal" axes of a point cloud distributed as a multivariate gaussian distribution. More precisely it looks for the axes which decorrelate the data components which correspond to the axes which carry the maximum variance. Thus, PCA can be seen as the solution of the following optimization problem:

<wiki:gadget url="http://mathml-gadget.googlecode.com/svn/trunk/ascii-gadget.xml" border="0" width="100%" up\_content="`max\_w  w'C w`"/>
submited to
<wiki:gadget url="http://mathml-gadget.googlecode.com/svn/trunk/ascii-gadget.xml" border="0" width="100%" up\_content="`w' w = 1`"/>


which means "find the unit vector w along which the variance is maximal. C stands for the covariance matrix:

<wiki:gadget url="http://mathml-gadget.googlecode.com/svn/trunk/ascii-gadget.xml" border="0" width="100%" up\_content="`C = 1/(N-1) sum\_{i=1}^N (x\_i-mu)(x\_i-mu)'= 1/(N-1)  X'X`"/>

where µ is the barycenter of the data samples and X is the (N x d) data matrix where each row is a d-dimensional centered data sample.

Applying the Lagrange multiplier technique leads to the following solution:

<wiki:gadget url="http://mathml-gadget.googlecode.com/svn/trunk/ascii-gadget.xml" border="0" width="100%" up\_content="`C w = lambda w`"/>

Which means that the axes carrying the maximal variance are the eigenvectors of the covariance matrix.

Let W be the (k x d) where the columns are the k eigenvectors of C sorted by decreasing order of the corresponding eigenvalue. One can reduce the dimensionality of the problem by projecting data samples on the principal axes using:

<wiki:gadget url="http://mathml-gadget.googlecode.com/svn/trunk/ascii-gadget.xml" border="0" width="100%" up\_content="`Y = XW`"/>

Each axis can be given the same importance by "whitening" the projection:

<wiki:gadget url="http://mathml-gadget.googlecode.com/svn/trunk/ascii-gadget.xml" border="0" width="100%" up\_content="`Y = XW Lambda^{-1}`"/>

<wiki:gadget url="http://mathml-gadget.googlecode.com/svn/trunk/ascii-gadget.xml" border="0" width="100%" up\_content="where `Lambda` is the diagonal matrix with the eigenvalues corresponding the eigenvectors in W"/>

## Kernel PCA ##

We derive the Kernel formulation of PCA using the SVD decomposition of X in the linear case. The matrix X can be decomposed as:

<wiki:gadget url="http://mathml-gadget.googlecode.com/svn/trunk/ascii-gadget.xml" border="0" width="100%" up\_content="`X' = USv'`"/>
and C is then:

<wiki:gadget url="http://mathml-gadget.googlecode.com/svn/trunk/ascii-gadget.xml" border="0" width="100%" up\_content="`C =  1/(N-1) X'X = 1/(N-1) v S^2 v' = W Lambda W'`"/>

We consider the eigen-decomposition of the external product which corresponds to  the kernel matrix associated with the linear kernel
:
<wiki:gadget url="http://mathml-gadget.googlecode.com/svn/trunk/ascii-gadget.xml" border="0" width="100%" up\_content="`K = XX' = US^2U'`"/>

The eigen vectors of K and those of C are related through
<wiki:gadget url="http://mathml-gadget.googlecode.com/svn/trunk/ascii-gadget.xml" border="0" width="100%" up\_content="`U = X'W S^{-1}`"/>

The projection equivalent to the non-kernel case is obtained by:
<wiki:gadget url="http://mathml-gadget.googlecode.com/svn/trunk/ascii-gadget.xml" border="0" width="100%" up\_content="`Y  = XU = XX'W S^{-1} = KWS^{-1}`"/>

And the whitened projection is:

The projection equivalent to the non-kernel case is obtained by:
<wiki:gadget url="http://mathml-gadget.googlecode.com/svn/trunk/ascii-gadget.xml" border="0" width="100%" up\_content="`Y  = XU = XX'W S^{-1} = sqrt(N-1) KWS^{-2}`"/>

# The PCA module #

The PCA module allows you to perform the basic operations described above, though the PCA object.

## Constructor ##

The constructor admits the following arguments:
  * k: the number of components to compute. The default is None which corresponds to all components. Be aware that if the Covariance matrix is not of full rank this might lead to some problems when computing the whiten PCA.

  * kernel: (default False) Do we perform Kernel PCA. If True the matrix given to the "fit" method is supposed to be centered.

  * extern: Perform PCA by computing the eigen vectors of the extern matrix product (linear kernel). This can speed-up the computations when then number of features is much bigger than the number of samples.

## Fit method ##

The eigen decomposition is done by calling the "fit" method. This methods takes as argument the data matrix X in the case of the usual PCA and the centered kernel matrix K in the case of Kernel PCA. Once the fit method has been called, the following attributes are available:

  * eigen\_values_: the computed eigenvalues sorted by decreasing order
  * eigen\_vectors_ : the corresponding eigenvectors where the i-th column corresponds to the i-th eigenvalue.
  * mean: the barycenter of the data samples (only for usual PCA)

## Transform method ##
The projection is performed by calling the "transform" method on the data matrix to project. This method takes as argument:

  * X: the data to project. For Kernel PCA, "fit" was called using a Kernel Matrix K corresponding to the kernel values between all pairs of samples. When one want to apply the projection to a new set of data, the matrix given to "transform" should correspond to the kernel values between the new samples (as rows) the samples used to compute the decomposition when calling "fit" (as columns).
  * whiten: (default False) Should the projection axes be whitened.

## Sample code ##

```
import PCA
pca = PCA.PCA(k = 3,extern = True) # perform a 3-dimensional PCA
pca.fit(X) # learn the projection using X
proj = pca.transform(Y) # Apply projection to Y

pca = PCA.PCA(k = 4,kernel = True) # perform a 4 dimensional kernel PCA.
pca.fit(Kxx) # learn the projection using Kx
proj = pca.transform(Kyx,whiten = True) # Project with whitening
```