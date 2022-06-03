import numpy as np
import random 

# a = np.random.randn(1, 3, 3)
# b = np.random.randn(1, 32, 3)

a = np.random.randn(32, 3, 3)
b = np.random.randn(32, 32, 3)

print(a)
print(b)

# c = np.matmul(a, b)
c = np.inner(a, b)

print(c)
print(c.shape)
