import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import probplot

from sklearn import datasets
from sklearn import linear_model
from sklearn import ensemble

########### Linear regression
# boston = datasets.load_boston()
# # print(boston)
# from sklearn.linear_model import LinearRegression
# # lr = LinearRegression()
# # print boston.data.shape, boston.target.shape
# # lr.fit(boston.data, boston.target)
# # predictions = lr.predict(boston.data)
# lr2 = LinearRegression(normalize=True)
# lr2.fit(boston.data, boston.target)
# predictions2 = lr2.predict(boston.data)
# for i in range(len(boston.feature_names)):
# 	print('%s=%s' % (boston.feature_names[i],lr2.coef_[i]))
# print(predictions2.shape)
# res = predictions2-boston.target

################ MSE
def MSE(target, predictions):
	squared_deviation = np.power(target-predictions, 2)
	return np.mean(squared_deviation)
############ MAD
def MAD(target, predictions):
	absolute_deviation = np.abs(target-predictions)
	return np.mean(absolute_deviation)
########### regularization with LARS
# from sklearn.datasets import make_regression
# reg_data, reg_target = make_regression(n_samples=200, n_features=500, n_informative=10, noise=2)
# print reg_data.shape, reg_target.shape

# from sklearn.linear_model import Lars
# lars = Lars(n_nonzero_coefs=10)
# lars.fit(reg_data, reg_target)
# print np.sum(lars.coef_!=0), lars.intercept_

# from sklearn.linear_model import LarsCV
# lcv = LarsCV()
# lcv.fit(reg_data, reg_target)
# print np.sum(lcv.coef_!=0), lcv.intercept_
############  Linear method for classification-logistic regression
# from sklearn.datasets import make_classification
# X,y = make_classification(n_samples=000, n_features=4)
# print X[:4], y[:4], set(y)
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression()
# X_train = X[:-200]
# X_test = X[-200:]
# y_train = y[:-200]
# y_test = y[-200:]
# lr.fit(X_train, y_train)
# y_train_predictions = lr.predict(X_train)
# y_test_predictions = lr.predict(X_test)
# print (y_train_predictions == y_train).sum().astype(float)/len(y_train)
# print (y_test_predictions == y_test).sum().astype(float)/len(y_test)
#############  Bayesian ridge regression
# from sklearn.linear_model import BayesianRidge
# X,y = datasets.make_regression(n_samples=1000, n_features=10, n_informative=2, noise=20)
# br = linear_model.BayesianRidge()
# br.fit(X, y)
# print br.coef_
# br_alphas = linear_model.BayesianRidge(alpha_1=10, lambda_1=10)
# br_alphas.fit(X, y)
# print br_alphas.coef_
#############  gradient boosting
# X,y = datasets.make_regression(1000, 2, noise=10)
# gbr = ensemble.GradientBoostingRegressor()
# gbr.fit(X, y)
# gbr_preds = gbr.predict(X)
# gbr_res = gbr_preds - y
# print np.power(gbr_res, 2).sum()
# print np.percentile(gbr_res, [2.5, 97.5])

# lr = linear_model.LinearRegression()
# lr.fit(X, y)
# lr_preds = lr.predict(X)
# lr_res = lr_preds - y
# print np.power(lr_res, 2).sum()
# print np.percentile(lr_res, [2.5, 97.5])









############# make a figure#####################
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
ax.hist(gbr_res)
ax.hist(lr_res)
ax.set_title('my plot')
plt.show()

# fig = plt.figure(figsize=(7,5))
# ax = fig.add_subplot(111)
# probplot(y, plot=ax)
# plt.show()