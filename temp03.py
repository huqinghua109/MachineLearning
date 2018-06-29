import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import cluster

######################Building models with distance metrics
blobs, classes = datasets.make_blobs(500, centers=3)
# print blobs, classes
kmean = cluster.KMeans(n_clusters=3)
kmean.fit(blobs)
print kmean
print kmean.cluster_centers_

########### make a picture
fig, ax = plt.subplots(figsize=(7.5,7.5))
rgb = np.array(['r', 'g', 'b'])
ax.scatter(blobs[:,0], blobs[:,1], color=rgb[classes])
ax.scatter(kmean.cluster_centers_[:,0], kmean.cluster_centers_[:,1], marker='*', s=250, color='black', label='Centers')
ax.set_title("blobs")
ax.legend(loc='best')
plt.show()