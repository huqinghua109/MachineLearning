from scipy.stats import probplot
import numpy as np
import matplotlib.pyplot as plt

########## Q-Q plot
x = np.random.standard_normal(10)
print(x)
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
probplot(x, plot=ax)
plt.show()