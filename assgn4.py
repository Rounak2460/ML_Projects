import numpy as np
train_y = []
class Node:
    def _init_(self,a):
        self.children = None
        self.value = a
    
    def remove(self, v):
        self.children.remove(self.children[v])
        return (self.children[v])
    
    def add(self, v):
        self.children.append(v)
        return (self)
        


def h(y):
    tot = len(y)
    a = np.count_nonzero(y)
    rich = a/tot
    poor = 1-rich
    ans = -rich*np.log2(rich)-poor*np.log2(poor)
    return ans




def Information_gain(x,y,attr):
    a  = x[: ,attr:attr+1]
    tot = len(a)
    b = np.unique(a)
    parent = h(y)
    z = []
    for i in range(len(b)):
        v = []
        su = 0
        for j in range(len(x)):
            if (x[j][attr] == b[i]):
                v.append(y[j])
                su = su + 1
        z.append(h(v),su)
    for k in range(len(z)):
        (hsv,sv) = z[k]
        parent = parent - (float(sv/tot))*hsv
    return parent
        




def split(x, y, attr):
    a  = x[: ,attr:attr+1]
    b = np.unique(a)
    z = []
    for i in range(len(b)):
        v = []
        for j in range(len(x)):
            if (x[j][attr] == b[i]):
                v.append(y[j])
        z.append(v)
    return z
                  




root = Node(train_y)                  
def dec_tree(train_x, train_y):
    v = train_x.shape
    temp = Node(train_y)
    n = 1
    z = []
    if (h(temp.value) == 0.0):
        return (root,n)
    for i in range(0,v[1]):
        z.append(Information_gain(train_x,temp.value, i))
    attr = z.index(max(z))
    temp.children = split(train_x, train_y, attr)
    for j in temp.children:
        dec_tree(train_x,j)
        n = n + 1
        

def pruning(train_x,train_y,val_x, val_y):
    decision = dec_tree(train_x,train_y)
    for i in range(len(decision.children)):
        decision.remove()
        
        
    
    
    
        

