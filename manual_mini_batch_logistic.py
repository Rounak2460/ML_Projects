import numpy as np
import csv
import sys
from scipy.special import softmax
train = sys.argv[1]
test = sys.argv[2]
output = sys.argv[3]
'''param = sys.argv[3]
out = sys.argv[4]
weight = sys.argv[5]'''

mat1 = np.genfromtxt(train, delimiter = ',', dtype = str)
mat2 = np.genfromtxt(test, delimiter = ',', dtype = str)


def one_hot(m):
    v = m.shape
    l = np.array([])
    for i in range(0, v[1]):
        k = m[ :, i:i+1]
        l = np.concatenate((l , (np.unique(k))), axis = 0)
    q  = l.shape[0]
    a = np.zeros((v[0],q))
    for i in range(v[0]):
        b = m[i:i+1, : ]
        for j in range(v[1]):
            o = 0
            while (b[0, j] != l[o]):
                o = o + 1
            a[i , o] = 1
    return (a)

def loss(train_x, w, y):
    sum = 0
    for i in range(train_x.shape[1]):
        sum = sum + ((y[i:i+1, : ]).dot((np.log(softmax(train_x.dot(w)[i:i+1, : ], axis = 1))).T))
    return (sum/2*train_x.shape[0])

def f(v):
    a = np.argmax(v, axis = 1)
    j = 0
    x = np.zeros(v.shape)
    for i in a:
        x[j][i] = 1
        j = j + 1
    return (x)

def logistic_k(train_x, y, test_x, p, eta):
    n = train_x.shape[0]
    w = np.zeros((train_x.shape[1], y.shape[1]))
    X_ = train_x[0: int(n/p), :]
    for i in range(0, n, int(n/p)):
        X_ = train_x[i: i + int(n/p), :]
        w = w - (X_.T).dot((softmax(X_.dot(w), axis = 1)- y[i: i + int(n/p), :]))*eta
    c = (softmax(train_x.dot(w), axis = 1))
    d = (softmax(test_x.dot(w), axis = 1))
    acc =0
    for i in range(c.shape[0]):
        if (y[i][(np.argmax(c[i:i+1, :]))] == 1):
            acc = acc + 1
    print(acc/(train_x.shape[0]))
    return (f(d))

one_x = np. ones((mat1.shape[0],1))
train_X = np.column_stack((one_x, one_hot(mat1[ : , 0: (mat1.shape[1]-1)])))
one_x1 = np. ones((mat2.shape[0],1))
test_X = np.column_stack((one_x1, one_hot(mat2)))
train_y =  one_hot(mat1[ : , (mat1.shape[1]-1): ])
feature1 = train_X**2
test_X = np.column_stack((test_X, feature1))
np.savetxt(output, logistic_k(train_X, train_y, test_X, 10, 0.01 ), delimiter = "\n")
