import csv
import numpy as np
import sys
train = sys.argv[1]
param = sys.argv[2]
weight = sys.argv[3]
def one_hot(train_y):
    a = np.array(range(0,10))
    matrx = np.zeros((train_y.shape[0],10))
    for i in range(train_y.shape[0]):
        ans = int(train_y[i][0])
        matrx[i][ans] = 1.0
    return(matrx)

def g(v):
    return 1./(1.+np.exp(-v))
def h(v):
    v1 = np.exp(v)/(np.sum(np.exp(v) , axis = 1))[: , None]
    return(v1)

def batches_(v,batch):
    n = len(v)
    k = n/batch
    x = []
    for i in range(0,len(v),batch):
        x.append(v[i:i+batch,])
    return(x)

def feed_frwrd(train_x,w,o,b):
    activ = []
    activ.append(train_x)
    for i in range(len(o)):
        w1 = w[i]
        b1 = b[i]
        z = activ[i].dot(w1) + b1
        activ.append(g(z))
    activ.append(h(activ[-1].dot(w[-1]) + b[-1]))
    return(activ)
#I have applied AndrewNg algorithm given on youtube for back propagation

def back_propgn(train_x,train_y,w,o,b):
    activate = feed_frwrd(train_x,w,o,b)
    deriv_w =  []
    deriv_b =  []
    deriv_z =  []
    deriv_a =  []
    deriv_z.append(np.divide((1-train_y) , (1-activate[-1])) - np.divide((train_y),activate[-1]))
    deriv_a.append(np.multiply(deriv_z[0], np.multiply((activate[-1]) , (1-activate[-1]))))
    for i in range(len(w)):
        h = ((activate[len(activate)-2-i].T).dot(deriv_a[i]))/len(train_x)
        deriv_w.append(h)
        deriv_b.append(np.sum(deriv_a[i]/len(train_x),axis = 0, keepdims = True))
        deriv_z.append(deriv_a[i].dot((w[len(w)-1-i]).T))
        deriv_a.append(np.multiply(deriv_z[i+1], np.multiply((activate[len(activate)-2-i]) , (1-activate[len(activate)-2-i]))))
    deriv_b.reverse()
    deriv_w.reverse()
    return([deriv_w,deriv_b])

def neural(train_x, train_y, v):
    operate = int(v[0])
    eta0 = float(v[1])
    iter1 = int(v[2])
    batch = int(v[3])
    p = train_x.shape
    o = v[4].split(" ")
    a = []
    a.append(train_x.shape[1])
    for i in range(len(o)):
        a.append(int(o[i]))
    a.append(train_y.shape[1])
    print(a)
    w = []
    b = []
    for i in range(len(a) - 1):
        w1 = np.zeros((a[i],a[i+1]))
        b1 = np.zeros((1,a[i+1]))
        w.append(w1)
        b.append(b1)
    k = p[0]/batch
    X_batch = batches_(train_x,batch)
    Y_batch = batches_(train_y,batch)
    
    for j in range(0,iter1):
        X = X_batch[int(j%k)]
        Y = Y_batch[int(j%k)]
        ans = back_propgn(X,Y,w,o,b)
        eta = eta0/(j+1)**0.5
        for l in range(len(w)):
            w[l] = w[l] - eta*ans[0][l]
            b[l] = b[l] - eta*ans[1][l]
        
    return([b,w])

mat1 = np.loadtxt(train,dtype='float',delimiter=',')
with open(param , 'r') as f:
    file = f.read()
f.close()
weights = open(weight, "a")
input_x = mat1[:,:-1]
input_y = one_hot(mat1[:,-1].reshape((-1,1)))
u = file.split('\n')
ans = neural(input_x, input_y, u)
bias = ans[0]
wts = ans[1]
for i in range(len(wts)):
    np.savetxt(weights, bias[i], delimiter = "\n")
    np.savetxt(weights, wts[i], delimiter = "\n")
