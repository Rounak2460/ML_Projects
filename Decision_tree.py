import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
train_file = sys.argv[1]
valid_file = sys.argv[2]
test_file = sys.argv[3]
val_pred_file = sys.argv[4]
test_pred_file = sys.argv[5]

train_df = pd.read_csv(train_file)
valid_df = pd.read_csv(valid_file)
test_df = pd.read_csv(test_file)

train_arr_data = train_df.values
valid_arr_data = valid_df.values
test_arr_data = test_df.values
colum = train_df.columns
feature_types = []
for i in range(len(colum)):
    if i != 14:
        if (isinstance(train_arr_data[0][i], str)):
            feature_types.append("categorical")
        else:
            feature_types.append("continous")
#print(feature_types)
#print(colum)
def classification(arr_data):
    labels = arr_data[:, -1]
    label, counting = np.unique(labels, return_counts = True)
    #print(label)
    #print(counting)
    b = np.argmax(counting)
    return (label[b])  
#print(classification(train_arr_data))
def splits_continous(arr_data, col_index, val):
    #column = arr_data[:, col_index]
    #print(type(arr_data))
    split1 = list([])
    split2 = list([])
    for i in range(len(arr_data)):
        if arr_data[i][col_index] <= val:
            split1.append(arr_data[i])
        else:
            split2.append(arr_data[i])
    split1 = np.asarray(split1)
    split2 = np.asarray(split2)
    return split1, split2

#s1, s2 = (splits_continous(train_arr_data, 4, np.median(train_arr_data[:, 4:5])))
#print(len(s1))
#print(len(s2))

def splits_discrete(arr_data, col_index):
    #print(type(arr_data))
    column = arr_data[:, col_index:col_index+1]
    unique_col = np.unique(column)
    v = []
    for i in unique_col:
        split1 = []
        for j in range(len(arr_data)):
            if arr_data[j][col_index] == i:
                split1.append(arr_data[j])
        v.append(np.asarray(split1))
    return v

'''v = splits_discrete(train_arr_data, 1)
k  = []
for i in v:
    print (len(i))
    k.append(len(i))
k = np.asarray(k)
print (len(train_arr_data))
print(k.sum())
print("hi")
print (len(v))
'''    

def entropy(arr_data):
    #print(type(arr_data))
    label = arr_data[:,len(arr_data[0])-1:len(arr_data[0])]
    _, counting = np.unique(label, return_counts = True)
    probab = counting/counting.sum()
    #print(probab)
    entrpy = 1-sum(probab*(probab))
    return (entrpy)

#print(entropy(train_arr_data))

def information_gain_continous(split1,split2):
    if (len(split2) == 0):
        return 0
    elif (len(split1) == 0):
        return 0
    else:
        net_data = np.concatenate((split1,split2), axis = 0)
        entrpy = entropy(net_data)
        p1 = len(split1)/(len(split1) + len(split2))
        p2 = len(split2)/(len(split1) + len(split2))
        entpy = p1*entropy(split1) + p2*entropy(split2)
        IG = entrpy - entpy 
        return (IG)

'''med = np.median(train_arr_data[:, 2:3])
print(med)
s1, s2 = (splits_continous(train_arr_data, 2, 178626.5))
print (information_gain_continous(s1,s2))'''

def information_gain_discrete(v):
    net_data = v[0]
    entropies = []
    for i in v:
        entropies.append(entropy(i))
    entropies = np.asarray(entropies)
    for i in range(1,len(v)-1):
        if (len(net_data) != 0 and len(v[i]) != 0):
            net_data = np.concatenate((net_data,v[i]), axis = 0)
        else:
            continue
    entrpy = entropy(net_data)
    g = []
    for i in range(len(v)):
        g.append(len(v[i]))
    g = np.asarray(g)
    prob = g/(g.sum())
    entpy = (prob*entropies).sum()
    IG = entrpy - entpy
    return (IG)
#v = splits_discrete(train_arr_data, 1)
#print(information_gain_discrete(v))

def best_split_median(arr_data):
    temp_ig = 0
    for i in range(len(colum)-1):
        if (feature_types[i] == "continous"):
           # print(i)
            #print(feature_types[i])
            #print(type(arr_data))
            #print("yes")
            #print(arr_data)
            col = arr_data[:,i:i+1]
            split_val = np.median(col)
            s1,s2 = splits_continous(arr_data, i, split_val)
            #print(s1,s2)
            ig = information_gain_continous(s1,s2)
            if (ig >= temp_ig):
                temp_ig = ig
                best_colm = i
                best_split = split_val
        else:
            #print("no")
            v = splits_discrete(arr_data, i)
            ig = information_gain_discrete(v)
            if (ig > temp_ig and len(v) != 1):
                temp_ig = ig
                best_colm = i
                best_split = "all"
    return best_colm, best_split

#print(best_split_median(train_arr_data))
#print(best_split_median(train_arr_data[:10,:]))

class Node:
    def __init__(self,arr_data):
        self.val = arr_data
        self.types = None
        self.split_value = None
        self.children = []
        self.split_col_index = None
        

#root = Node(train_arr_data)
def decision_tree(node, depth, sample_size, count = 0):
    arr_data = node.val
    #print(count)
    #print(type(arr_data))
    #print(count)
    col_ind, split_val = best_split_median(arr_data)
    if split_val != "all":
        split1, split2 = splits_continous(arr_data, col_ind, split_val)
        if (information_gain_continous(split1,split2) <= 0 or count >= depth or len(node.val) <= sample_size):
            return node
        else:
            #print(count)
            count += 1
            node1 = Node(split1)
            node2 = Node(split2)
            node.children.append(decision_tree(node1, depth,sample_size, count = count))
            node.children.append(decision_tree(node2, depth,sample_size, count = count))
            node.split_value = split_val
            #print(split_val)
            node.types = "continous"
            node.split_col_index = col_ind
            return node
    else:
        v = splits_discrete(arr_data, col_ind)
        if (information_gain_discrete(v) <= 0 or count >= depth or len(node.val) <= sample_size):
            return node
        else:
            count += 1
            for i in v:
                node1 = Node(i)
                node.children.append(decision_tree(node1,depth, sample_size,count = count))
                node.types = "discrete"
                #print(i[0][col_ind])
                node.split_value = i[0][col_ind]
                node.split_col_index = col_ind
            return node

#print(len((our_tree.children[0]).children))
#print(our_tree.val) 
def view_tree(tree):
    if (len(tree.children) == 0): 
        #print("hi")
        return(str(tree.split_col_index) + "," +  str(tree.split_value) + "\n")
        
    else:
        #print("hello")
        #print(str(tree.split_col_index) + "," + str(tree.split_value) +  "\n")
        s = ""
        for i in tree.children:
            s = s + view_tree(i) + "  "
        return(s + "\n")
#print(len(our_tree.children))
#print(our_tree.split_col_index)
#print(view_tree(our_tree))
def classify_new(eg, tree, arr_data = train_arr_data):
    if (len(tree.children) == 0):
       return classification(tree.val)
    else:
        #print("no")
        col_index = tree.split_col_index
        #print(eg[col_index])
        #print(tree.types)
        if tree.types == "continous":
            split_val = tree.split_value
            if (eg[col_index] <= split_val):
               # print(arr_data[:,col_index],split_val)
                return classify_new(eg,tree.children[0])
            else:
                return classify_new(eg,tree.children[1])
        else:
            #print("hi")
            #print(len(tree.children))
            for w in tree.children:
                g_arr = w.val
                if (eg[col_index] == g_arr[0][col_index]):
                    #print("hey")
                    return classify_new(eg,w)

def accuracy(arr_data, tree):
    feature1 = arr_data[:, :len(arr_data[0])-1]
    label = arr_data[:, len(arr_data[0])-1:len(arr_data[0])]
    prediction = np.zeros((len(arr_data),1))
    for i in range(len(feature1)):
        prediction[i] = classify_new(feature1[i], tree)
    correct = (label == prediction)
    acc = correct.mean()
    return acc

           
def prediction(arr_data, tree):
    feature1 = arr_data[:, :len(arr_data[0])-1]
    predictions = np.zeros((len(arr_data),1))
    for i in range(len(feature1)):
        predictions[i] = classify_new(feature1[i], tree)
    return predictions

#print(prediction(valid_arr_data,our_tree))

def pruning(tree,arr_data):
    #print("hi",len(tree.children))
    #print(len(tree.val))
    if (len(tree.children) == 0 or len(arr_data) == 0):
        #print("here",len(tree.children))
        return tree
    else:
        col_index = tree.split_col_index
        if tree.types == "continous":
            print("yes")
            split_val = tree.split_value
            x = (valid_arr_data[np.nonzero(valid_arr_data[:,col_index]<=split_val)])
            y = (valid_arr_data[np.nonzero(valid_arr_data[:,col_index]>split_val)])
            tree.children[0] = pruning(tree.children[0],x)
            tree.children[1] = pruning(tree.children[1],y)
            temp_acc = accuracy(arr_data,tree)
            a = tree.children
            tree.children = []
            acc = accuracy(arr_data, tree)
            if (acc > temp_acc):
                return tree
            else:
                tree.children = a
                return tree
        else:
            print("no")
            count = 0
            #print(arr_data)
            for i in tree.children:
                #print(count)
                split_val = tree.split_value
                #print(valid_arr_data[:,col_index],'x'+split_val)
                #print(np.nonzero(valid_arr_data[:,col_index]==split_val))
                y = (valid_arr_data[np.nonzero(valid_arr_data[:,col_index]==split_val)])
                #print(y)
                tree.children[count] = pruning(tree.children[count],y)
                count += 1
            #print(arr_data)
            temp_acc =  accuracy(arr_data, tree)
            a = tree.children
            tree.children = []
            #print(a)
            acc = accuracy(arr_data, tree)
            if (acc > temp_acc):
                return tree
            else:
                tree.children = a
                return tree
                    
                       
            
def number_of_nodes(tree, count =1):
    if len(tree.children) == 0 :
        return count
    else:
        print(count)
        for i in tree.children:
            count = count + 1
            number_of_nodes(i,count = count)
#print(number_of_nodes(our_tree))
        
    
#pruned_tree = pruning(our_tree,valid_arr_data)
node1 = Node(train_arr_data)
our_tree =  decision_tree(node1,6,330)
v1 = prediction(valid_arr_data, our_tree)
v2 = prediction(test_arr_data, our_tree)
print(accuracy(valid_arr_data, our_tree))
out1 = open(val_pred_file, "a")  
out2 = open(test_pred_file, "a")  

np.savetxt(val_pred_file, v1, delimiter = "\n")  
np.savetxt(test_pred_file, v2, delimiter = "\n")          

