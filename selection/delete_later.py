import numpy as np
import random 

a = np.random.randn(2, 3072)
b = np.random.randn(1, 3072)

inner = np.inner(a, b).reshape(-1)

print(inner.shape)



# a = np.random.randn(1, 3072)

# print(a.shape)

# print(a[0])


# print(a)
# print(b)

# c = np.matmul(a, b)
# c = np.inner(a, b)

# print(c)
# print(c.shape)

quit()

feat = np.random.randn(32, 32)
b = feat / np.linalg.norm(feat)
# b = b.reshape(-1, 3072)
h = np.random.randn(32, 32)
# h = h.reshape(-1, 3072)

d, e, f = np.linalg.svd(h)

print(h)

print()
print("VH")
print(f)
print()
print("V")
print(d)
print()
print("Eigen Values")
print(e)

# print(d)
# print(e)
print(d.shape, e.shape, f.shape)

print(f[0].shape)


# print(d[:][0][:].shape)

ff = np.inner(b, f[0])

print(ff)
print(ff.shape)

quit()

gg = np.cross()

print()
print()

a1 = np.random.randn(2, 3)
a2 = np.random.randn(3, 3)

a3 = np.inner(a1, a2)

print(a3.shape)

# ff = np.inner(f[0].reshape(-1, 3072), b.reshape(-1, 3072))

# print(ff[0])
# print(ff[0].shape)

import numpy as np
 
def PCA(X , num_components):
     
    #Step-1
    X_meaned = X - np.mean(X , axis = 0)
     
    #Step-2
    cov_mat = np.cov(X_meaned , rowvar = False)
     
    #Step-3
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    #Step-5
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
    #Step-6
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
    return X_reduced

# asd = PCA(h.reshape(-1, 3072), 1)
# print(asd.shape)

# ff1 = np.inner(d[0].reshape(-1, 3072), b.reshape(-1, 3072))

# print(ff1[0])
# print(ff1[0].shape)

# tt = np.matmul(d, np.matmul(e, f))

# print(tt.shape)
