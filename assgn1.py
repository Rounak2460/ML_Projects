import numpy as np
import csv
import sys
from io import StringIO
from sklearn import linear_model
import math

def mp_inverse(x,y,test_x):
     n = np.ones ((x.shape[0],1))
     o = np.ones ((test_x.shape[0],1))
     q = np.column_stack((n, x))
     r = np.column_stack((o, test_x))
     w = ((np.linalg.inv((q.T).dot(q))).dot(q.T)).dot(y)
     z = (r.dot(w))
     loss = (np.linalg.norm((q.dot(w) - y)))**2/(2*x.shape[0])
     return ([z,w])


def lamda (x,y,lmda):
  mean = []
  n = x.shape[0]
  for j in lmda:
      error = 0
      for i in range(0,n,int(n/10)):
        valid_x = x[i: i+int(n/10), : ]
        valid_y = y[i: i+int(n/10)]
        train_x = np.delete(x, range(i,i+int(n/10)), axis =0)
        train_y = np.delete(y, range(i,i+int(n/10)), axis =0)
        one_x = np.ones((train_x.shape[0],1))
        X = np.column_stack((one_x,train_x))
        oneval_x = np.ones((valid_x.shape[0],1))
        valid_x1 = np.column_stack((oneval_x,valid_x))
        w = ((np.linalg.inv((X.T).dot(X) + np.eye(X.shape[1])*j)).dot(X.T)).dot(train_y)
        error = error  + (np.linalg.norm(valid_x1.dot(w)- valid_y))**2
      mean.append(error)
  min_ = 0
  for k in range(len(mean)):
    if (mean[k] < mean[min_] ):
      min_ = k
  return(lmda[min_])

def ridge_reg(x, y, test_x, lmda):
  l = lamda(x,y,lmda)
  n = x.shape[0]
  one_x = np.ones((x.shape[0],1))
  X = np.column_stack((one_x, x))
  w = ((np.linalg.inv((X.T).dot(X) + np.eye(X.shape[1])*l)).dot(X.T)).dot(y)
  onetest_x = np.ones((test_x.shape[0],1))
  r = np.column_stack((onetest_x, test_x))
  z = (r.dot(w))
  return ([z,w])


def lasso(X,y,test_X):
    mean = []
    lmda = [0.001,0.003,0.01,0.016,0.02]
    feature1 = np.exp(np.absolute(X[ : ])*-1)
    sigma = 0.1
    test_feat = np.exp(np.absolute(test_X[ :])*-1)
    x = np.concatenate((X,feature1),axis= 1)
    test_x = np.concatenate((test_X,test_feat),axis= 1)
    n = x.shape[0]
    for j in lmda:
        error = 0
        for i in range(0,n,int(n/10)):
            valid_x = x[i: i+int(n/10), : ]
            valid_y = y[i: i+int(n/10)]
            train_x = np.delete(x, range(i,i+int(n/10)), axis =0)
            train_y = np.delete(y, range(i,i+int(n/10)), axis =0)
            reg = linear_model.LassoLars(alpha = j)
            reg.fit(train_x,train_y)
            w = reg.coef_
            error = error  + (np.linalg.norm(valid_x.dot(w)- valid_y))**2
        mean.append(error)
    min_ = 0
    for k in range(len(lmda)):
        if (mean[k] < mean[min_] ):
            min_ = k
    return(lmda[min_])

def lasso_reg(X, y, test_X):
    l = lasso(X,y,test_X)
    feature1 = np.exp(np.absolute(X[ : ])*-1)
    test_feat = np.exp(np.absolute(test_X[ :])*-1)
    x = np.concatenate((X,feature1),axis= 1)
    test_x = np.concatenate((test_X,test_feat),axis= 1)
    n = x.shape[0]
    reg = linear_model.LassoLars(alpha = l)
    reg.fit(x,y)
    w = reg.coef_
    z = (test_x.dot(w))
    return (z)


prgrm = sys.argv[1]
if (prgrm == 'a'):
    training = sys.argv[2]
    testing = sys.argv[3]
    output = sys.argv[4]
    weights = sys.argv[5]
    mat1 = np.genfromtxt(training, delimiter = ",")
    mat2 = np.genfromtxt(testing, delimiter = ",")
    input_x = mat1[ : , :(mat1.shape[1]-1)]
    input_y = mat1[: , (mat1.shape[1]-1): ]
    np.savetxt(output, mp_inverse(input_x, input_y, mat2)[0], delimiter = "\n")
    np.savetxt(weights, mp_inverse(input_x, input_y, mat2)[1], delimiter = "\n")
if (prgrm == 'b'):
    training = sys.argv[2]
    testing = sys.argv[3]
    regular = sys.argv[4]
    output = sys.argv[5]
    weights = sys.argv[6]
    mat1 = np.genfromtxt(training, delimiter = ",")
    lmda = np.genfromtxt(regular, delimiter = ",")
    mat2 = np.genfromtxt(testing, delimiter = ",")
    input_x = mat1[ : , :(mat1.shape[1]-1)]
    input_y = mat1[: , (mat1.shape[1]-1): ]
    np.savetxt(output, ridge_reg(input_x, input_y,mat2, lmda)[0], delimiter = "\n")
    np.savetxt(weights, ridge_reg(input_x, input_y, mat2,lmda)[1], delimiter = "\n")
if (prgrm == 'c'):
    training = sys.argv[2]
    testing = sys.argv[3]
    output = sys.argv[4]
    mat1 = np.genfromtxt(training, delimiter = ",")
    mat2 = np.genfromtxt(testing, delimiter = ",")
    input_x = mat1[: , :mat1.shape[1]-1]
    input_y = mat1[:, (mat1.shape[1]-1): ]
    np.savetxt(output, lasso_reg(input_x, input_y,mat2), delimiter = "\n")
