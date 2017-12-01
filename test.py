import numpy as np
from scipy import sparse

# read file
m1 = np.loadtxt('/Users/kaixuangao/Desktop/Sample.txt')
# m1 = np.loadtxt('/Users/kaixuangao/Downloads/AdjacencyMatrix.txt')
col = m1[:, 0] - 1
row = m1[:, 1] - 1
data = m1[:, 2]
scale = int(m1.max())
m2 = sparse.coo_matrix((data, (row, col)), shape=(scale, scale)).tocsc()

s = np.array(m2.sum(axis=0))
for i in range(len(s[0])):
    if s[0][i] != 0:
        for j in range(m2.indptr[i], m2.indptr[i + 1]):
            m2.data[j] /= s[0][i]

print('Matrix M in dense format: ')
print(m2.todense())
print('\n')

print('Matrix M in coo format: ')
print(m2)
print('\n')

rj = np.ones((scale, 1)) / scale

print('Original rank vector:')
print(rj)
print('\n')

beta = 0.85
nums_itr = 0
while True:
    c_rank = rj
    nums_itr += 1
    rj = beta * m2.dot(rj) + (1 - beta) * 1/scale
    flag = 1
    check = rj - c_rank
    for i in check:
        if abs(i) > 0.0001:
            flag = 0
    if flag == 1:
        break

print('Converged rank vector: ')
print(rj)
print('\n')

print('Numbers of iterations:')
print(nums_itr)
print('\n')


