import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import probplot

from sklearn import datasets
########### Linear regression
boston = datasets.load_boston()
# print(boston)
from sklearn.linear_model import LinearRegression
# lr = LinearRegression()
# print boston.data.shape, boston.target.shape
# lr.fit(boston.data, boston.target)
# predictions = lr.predict(boston.data)
lr2 = LinearRegression(normalize=True)
lr2.fit(boston.data, boston.target)
predictions2 = lr2.predict(boston.data)
for i in range(len(boston.feature_names)):
	print('%s=%s' % (boston.feature_names[i],lr2.coef_[i]))
print(predictions2.shape)
res = predictions2-boston.target

################ MSE
def MSE(target, predictions):
	squared_deviation = np.power(target-predictions, 2)
	return np.mean(squared_deviation)
############ MAD
def MAD(target, predictions):
	absolute_deviation = np.abs(target-predictions)
	return np.mean(absolute_deviation)
# fig, ax = plt.subplots()
# ax.hist(res)
# ax.set_title('res hist')
# plt.show()

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
probplot(res, plot=ax)
plt.show()