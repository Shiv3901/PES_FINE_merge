import numpy as np
import random 

# a = np.random.randn(1, 3, 3)
# b = np.random.randn(1, 32, 3)

a = np.random.randn(32, 3, 3)
b = np.random.randn(32, 32, 3)

# print(a)
# print(b)

# c = np.matmul(a, b)
# c = np.inner(a, b)

# print(c)
# print(c.shape)

h = np.random.randn(500, 32, 32, 3)

d, e, f = np.linalg.svd(h, full_matrices=False)

# print(d)
# print(e)
print(d.shape, e.shape, f.shape)
print(d[0].shape)

ff = np.inner(d[0].reshape(-1, 3072), b.reshape(-1, 3072))

print(ff[0])
print(ff[0].shape)

# tt = np.matmul(d, np.matmul(e, f))

# print(tt.shape)
