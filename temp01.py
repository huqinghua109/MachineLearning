import numpy as np 
import pandas as pd 
import numpy as np
import scipy
import datetime as dtt
from sklearn import datasets
from sklearn import preprocessing

############### data in datasets 
# boston = datasets.load_boston()
# print type(boston)
# housing = datasets.fetch_california_housing()
# print type(housing)
# X,y = boston.data, boston.target
# print shape(X), len(y)
################ data that is maked
# complex_reg_data = datasets.make_regression(1000,10,5,2,1.0)
# print complex_reg_data[0].shape
# print len(complex_reg_data)
##############classification data
# classification_set = datasets.make_classification(weights=[0.1])
# print type(classification_set), len(classification_set[1])
##############cluster data
# blobs = datasets.make_blobs()
# print type(blobs), len(blobs[1])
################## sparse data 
# matrix = scipy.sparse.eye(100)
# print matrix
#################binarize data
# new_target = preprocessing.binarize(boston.target, threshold=boston.target.mean())
# print new_target,type(new_target)
# print boston.target[:5] > boston.target.mean()
##### mybin = preprocessing.Binarizer(boston.target.mean())
# new_target = mybin.fit_transform(boston.target)
# print new_target[:5]
##############sparse matrices
# spar = scipy.sparse.coo.coo_matrix(np.random.binomial(1, 0.25, 100))
###############categorical variables
iris = datasets.load_iris()
iris_X = iris.data
y = iris.target
# d = np.column_stack((X, y))
# print len(d)
# print d[0]
# text_encoder = preprocessing.OneHotEncoder()
# print text_encoder.fit_transform(d[:,-1:]).toarray()[:5]
# print text_encoder.fit_transform(np.zeros((3,1))).toarray()

############DictVectorizer
# from sklearn.feature_extraction import DictVectorizer
# dv = DictVectorizer()
# print y
# my_dict = [{'species':iris.target_names[i]} for i in y]
# print my_dict
# print dv.fit_transform(my_dict)
######patsy module
# import patsy
# print patsy.dmatrix("0 + C(species)", {'species':iris.target})
########Binarizing label features
# from sklearn.preprocessing import LabelBinarizer
# label_binarizer = LabelBinarizer()
# new_target = label_binarizer.fit_transform(y)
# print new_target.shape, y.shape
# print y
# print new_target
##########deal with Missing data & Imputer
# masking_array = np.random.binomial(1, 0.25, X.shape).astype(bool)
# print masking_array, X
# imputer = preprocessing.Imputer()
# iris_X_
############Pipeline
# mat = datasets.make_spd_matrix(10)
# # print mat, mat.shape, type(mat)
# masking_array = np.random.binomial(1, 0.1, mat.shape).astype(bool)
# # print masking_array
# mat[masking_array] = np.nan
# # print mat[0]
# imputer = preprocessing.Imputer()
# scaler = preprocessing.StandardScaler()
# mat_imputed = imputer.fit_transform(mat)
# # print mat_imputed.mean(), mat_imputed.std()
# mat_imp_scal = scaler.fit_transform(mat_imputed)
# # print mat_imp_scal.mean(), mat_imp_scal.std()
# from sklearn import pipeline
# pipe = pipeline.Pipeline([('impute',imputer), ('scaler', scaler)])
# print pipe
# new_mat = pipe.fit_transform(mat)
# print new_mat.mean(), new_mat.std()

###############Principal component analysis
from sklearn import decomposition
# pca = decomposition.PCA()
# print pca
# iris_pca = pca.fit_transform(iris_X)
# print pca.explained_variance_ratio_.sum()

# # print iris_X.shape,iris_X.mean(), iris_X.std()
# # print np.cov(iris_X.T)
# # print np.corrcoef (iris_X.T)
# # print '-'*20
# # print iris_pca.shape,iris_pca.mean(),iris_pca.std()
# # print np.corrcoef(iris_pca.T)
# pca = decomposition.PCA(n_components=2)
# # print pca
# iris_X_prime = pca.fit_transform(iris_X)
# print iris_X_prime.shape
# print pca.explained_variance_ratio_.sum()
#####factorAnalysis
# fa = decomposition.FactorAnalysis(n_components=2)
# iris_2_dim = fa.fit_transform(iris_X)
# print iris_2_dim.shape
# print iris_2_dim[0]
# print iris_X[0]
######### KernelPCA
# kpca = decomposition.KernelPCA(kernel='cosine', n_components=1)
# # print kpca
# A1_mean = [1,1]
# A1_cov = [[2,0.99], [1,1]]
# A1 = np.random.multivariate_normal(A1_mean, A1_cov, 50)
# A2_mean = [5,5]
# A2_cov = [[2,0.99], [1,1]]
# A2 = np.random.multivariate_normal(A2_mean, A2_cov, 50)
# A = np.vstack((A1,A2))

# B_mean = [5,0]
# B_cov = [[0.5, -1], [-0.9, 0.5]]
# B = np.random.multivariate_normal(B_mean, B_cov)

# AB = np.vstack((A,B))
# AB_transformed = kpca.fit_transform(AB)
# print B
######### SVD singular value decomposition
# svd = decomposition.TruncatedSVD()
# # print svd
# iris_transformed = svd.fit_transform(iris_X)
# print iris_X.shape, iris_transformed.shape
#####DictionaryLearning
dl = decomposition.DictionaryLearning(3)
transformed = dl.fit_transform(iris_X[::2])
print transformed.shape, iris_X[::2].shape
print transformed










