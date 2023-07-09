## This is course material for Introduction to Python Scientific Programming
## Example code: matplotlib_histogram.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

from matplotlib import patches
import numpy as np
import matplotlib.pyplot as plt

text = "We can know only that we know nothing. And that is the highest degree of human wisdom."
hist_labels = list('abcdefghijklmnopqrstuvwxyz') # label the 26 histogram bins with their text

data = np.zeros(0, dtype = int)
for c in text:
    if c.isalpha():
        index = int(ord(c.lower()) - ord('a'))
        data = np.append(data, index)
print(data)

max_value = np.max(data)
n, bins, patches = plt.hist(data, max_value+1, facecolor = 'green', alpha = 0.5)
plt.title("ASCII histogram")
label_axis = np.arange(0.5,26.5).tolist()
plt.xticks(label_axis, hist_labels)
plt.xlim(0, 26)
plt.show()