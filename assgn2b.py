import numpy as np
import csv
import sys
train = sys.argv[1]
test = sys.argv[2]
param = sys.argv[3]
output = sys.argv[4]
weights = sys.argv[5]


mat1 = np.genfromtxt(train, delimiter = ',', dtype = str)
mat2 = np.genfromtxt(test, delimiter = ',', dtype = str)
with open(param , 'r') as f:
    file = f.read()
f.close()

def master(m):
    v = m.shape
    l = np.array([])
    for i in range(0, v[1]):
        k = m[ :, i:i+1]
        l = np.concatenate((l , (np.unique(k))), axis = 0)
    return(l)
def softmax(v):
    v = np.exp(v)/(np.sum(np.exp(v) , axis = 1))[: , None]
    return(v)

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
        sum = sum + ((y[i:i+1, : ]).dot((np.log(softmax(train_x[i:i+1, : ].dot(w)))).T))
    return (sum/2*train_x.shape[0])

def f(v):
    a = np.argmax(v, axis = 1)
    j = 0
    x = np.zeros(v.shape)
    for i in a:
        x[j][i] = 1
        j = j + 1
    return (x)

def logistic_k(train_x, y, test_x, v):
    if (v[0] == "1"):
        eta = float(v[1])
        iter = int(v[2])
        n = train_x.shape
        p = int(n[0]/int(v[3]))
        k = y.shape
        w2 = np.zeros((n[1], k[1]))
        w1 = np.ones((n[1], k[1]))*7000000000000
        j = 0
        while(j<iter) > 10**-6):
            w1 = w2
            for i in range(0, n[0], int(n[0]/p)):
                X_ = train_x[i: i + int(n[0]/p), :]
                w2 = w2 - (X_.T).dot((softmax(X_.dot(w2))- y[i: i + int(n[0]/p), :]))*eta
                j = j + 1
        c = (softmax(train_x.dot(w2)))
        d = (softmax(test_x.dot(w2)))
        acc = 0
        for i in range(c.shape[0]):
            if (y[i][(np.argmax(c[i:i+1, :]))] == 1):
                acc = acc + 1
        print(j)
        print(acc/(train_x.shape[0]))
    elif (v[0] == "2"):
        eta = float(v[1])
        iter = int(v[2])
        n = train_x.shape
        p = int(n[0]/int(v[3]))
        k = y.shape
        w2 = np.zeros((n[1], k[1]))
        w1 = np.ones((n[1], k[1]))*7000000000000
        j = 0
        while(j<iter) > 10**-6):
            w1 = w2
            for i in range(0, n[0], int(n[0]/p)):
                X_ = train_x[i: i + int(n[0]/p), :]
                w2 = w2 - (X_.T).dot((softmax(X_.dot(w2))- y[i: i + int(n[0]/p), :]))*eta
                j = j + 1
                eta = eta/j**0.5
        c = (softmax(train_x.dot(w2)))
        d = (softmax(test_x.dot(w2)))
        acc = 0
        for i in range(c.shape[0]):
            if (y[i][(np.argmax(c[i:i+1, :]))] == 1):
                acc = acc + 1
        print(j)
        print(acc/(train_x.shape[0]))
    elif (v[0] == "3"):
        a_b = (v[1]).split(",")
        eta = float(a_b[0])
        n = train_x.shape
        alpha = float(a_b[1])
        beta = float(a_b[2])
        iter = int(v[2])
        p = int(n[0]/int(v[3]))
        k = y.shape
        w1 = np.ones((n[1], k[1]))*7000000000000
        j = 0
        w2 = np.zeros((n[1], k[1]))
        while (j < iter):
            d = (train_x.T.dot(y -(softmax(train_x.dot(w1))))).reshape(n[1]*k[1], 1)
            gradient = (train_x.T.dot(y -(softmax(train_x.dot(w1))))).reshape(n[1]*k[1], 1)
            if (loss(train_x, w2, y) >= loss(train_x, w1, y) + eta*alpha*(d.T.dot(gradient))):
                eta = eta*beta
            w1 = w2
            for i in range(0, n[0], int(n[0]/p)):
                X_ = train_x[i: i + int(n[0]/p), :]
                w2 = w2 - (X_.T).dot((softmax(X_.dot(w2))- y[i: i + int(n[0]/p), :]))*eta
                j = j + 1
            acc = 0
            c = (softmax(train_x.dot(w2), axis = 1))
            d = (softmax(test_x.dot(w2), axis = 1))
            for i in range(c.shape[0]):
                if (y[i][(np.argmax(c[i:i+1, :]))] == 1):
                    acc = acc + 1
        print(j)
        print(acc/(train_x.shape[0]))
    return ([f(d),w2])
one_x = np. ones((mat1.shape[0],1))
train_X = np.column_stack((one_x, one_hot(mat1[ : , 0: (mat1.shape[1]-1)])))
one_x1 = np. ones((mat2.shape[0],1))
test_X = np.column_stack((one_x1, one_hot(mat2)))
train_y =  one_hot(mat1[ : , (mat1.shape[1]-1): ])
t =  master(mat1[ : , (mat1.shape[1]-1): ])
u = file.split('\n')
r = (logistic_k(train_X, train_y, test_X, u))[0]
v = []
for p in range(r.shape[0]):
    v.append(t[np.argmax(r[p])])
np.savetxt(output, v, delimiter = "\n", fmt = "%s")
np.savetxt(weights, (logistic_k(train_X, train_y, test_X, u)[1]), delimiter = ",")
