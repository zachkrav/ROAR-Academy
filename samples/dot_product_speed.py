## This is course material for Introduction to Python Scientific Programming
## Example code: dot_product_speed.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import numpy as np
from time import time

test_dimension = 1000000

# Generate two long-dimension lists of all elements 1 and 2
L1 = [1.0]*test_dimension
L2 = [2.0]*test_dimension

# Benchmark the time for one dot product
begin_time = time()
result = sum(i[0]*i[1] for i in zip(L1, L2))
elapsed_time = time() - begin_time
print('List takes {0}s to compute {1}'.format(elapsed_time, result))

# Generate two long-dimension ndarrays of all elements 1 and 2
A1 = np.ones(test_dimension)
A2 = 2*np.ones(test_dimension)

# Benchmark the time for one dot product
begin_time = time()
result = A1.dot(A2)
elapsed_time = time() - begin_time
print('Numpy takes {0}s to compute {1}'.format(elapsed_time, result))