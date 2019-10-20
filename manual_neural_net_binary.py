import csv
import numpy as np
import sys
train = sys.argv[1]
param = sys.argv[2]
weight = sys.argv[3]
def g(v):
    return 1/(1+np.exp(-1*v))

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
    for i in range(len(o)+1):
        w1 = w[i]
        b1 = b[i]
        z = activ[i].dot(w1) + b1
        activ.append(g(z))
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
    eta = float(v[1])
    iter = int(v[2])
    batch = int(v[3])
    p = train_x.shape
    o = v[4].split(" ")
    print(o)
    a = []
    a.append(train_x.shape[1])
    for i in range(len(o)):
        a.append(int(o[i]))
    a.append(train_y.shape[1])
    w = []
    b = []
    for i in range(len(a) - 1):
        w1 = np.zeros((a[i],a[i+1]))
        b1 = np.zeros((1,a[i+1]))
        w.append(w1)
        b.append(b1)
    X_batch = batches_(train_x,batch)
    Y_batch = batches_(train_y,batch)
    k = len(train_x)/batch
    if (operate == 1):
        for j in range(iter):
            X = X_batch[int(j%k)]
            Y = Y_batch[int(j%k)]
            ans = back_propgn(X,Y,w,o,b)
            for l in range(len(w)):
                w[l] = w[l] - eta*ans[0][l]
                b[l] = b[l] - eta*ans[1][l]
    else:
        for j in range(iter):
            X = X_batch[int(j%k)]
            Y = Y_batch[int(j%k)]
            ans = back_propgn(X,Y,w,o,b)
            for l in range(len(w)):
                w[l] = w[l] - eta*ans[0][l]
                b[l] = b[l] - eta*ans[1][l]
                eta = eta/(j+1)**0.5

    return([b,w])

mat1 = np.genfromtxt(train, delimiter = ',')
with open(param , 'r') as f:
    file = f.read()
f.close()
weights = open(weight, "a")
input_x = mat1[: , :mat1.shape[1]-1]
input_y = mat1[:, (mat1.shape[1]-1): ]
u = file.split('\n')
print(u)
ans = neural(input_x, input_y, u)
bias = ans[0]
wts = ans[1]
for i in range(len(wts)):
    np.savetxt(weights, bias[i], delimiter = "\n")
    np.savetxt(weights, wts[i], delimiter = "\n")
