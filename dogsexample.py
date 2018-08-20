# Depicting the DATA

import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labradogs = 500

gray_height = 28 + 4*np.random.randn(greyhounds)
lab_height = 24 + 4*np.random.randn(labradogs)

plt.hist([gray_height,lab_height],stacked = True , color = ['r','b'])
plt.show()