import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from collections import deque

#train_file = sys.argv[1]
#valid_file = sys.argv[2]
#test_file = sys.argv[3]
#val_pred_file = sys.argv[4]
#test_pred_file = sys.argv[5]
train_file = 'train.csv'
valid_file = 'valid.csv'
#test_file = 'test_public.csv'
#val_pred_file = sys.argv[4]
#test_pred_file = sys.argv[5]

train_df = pd.read_csv(train_file)
valid_df = pd.read_csv(valid_file)
#test_df = pd.read_csv(test_file)

train_arr_data = train_df.values

def feature_type(data):
    types = []
    max_val = 15
    for features in data.columns:
        if features == "Rich?":
            continue
        else:
            unique = (train_df[features]).unique()
            val = len(unique)
            if (val > max_val, isinstance((train_df[features])[0], str)):
         #   if (isinstance((train_df[features])[0], str)):
                types.append("categorical")
            else:
                types.append("continous")

    #print (data.columns)
    return types


def is_same(arr_data):
    labels = arr_data[:, -1]
    a = np.unique(labels)
    if len(a) == 1:
        return True
    else:
        return False


def classification(arr_data):
    labels = arr_data[:, -1]
    label, counting = np.unique(labels, return_counts = True)
    b = np.argmax(counting)
    #print(counting)
    #print(label)
    return (label[b])
#print(classification(arr_data))


def splits_poss(arr_data):
    col = arr_data.shape[1]
    splitin = {}
    for i in range(col-1):
        splitin[i] = np.unique(arr_data[:, i])
    return (splitin)


def splits(data, col_index, val):
    arr_data = data.values
    column = arr_data[:,col_index]
    ftr_typ = feature_type(data)[col_index]
    if (ftr_typ == "continous"):
        split1 = arr_data[column<=val]
        split2 = arr_data[column>val]
    else:
        split1 = arr_data[column==val]
        split2 = arr_data[column!=val]
    return split1, split2


def entropy(arr_data):
    label = arr_data[:,-1]
    _, counting = np.unique(label, return_counts = True)
    probab = counting/counting.sum()
    entrpy = -sum(probab*np.log2(probab))
    return (entrpy)

def entropy_gini(arr_data):
    label = arr_data[:,-1]
    _, counting = np.unique(label, return_counts = True)
    probab = counting/counting.sum()
    entrpy = 1-sum(probab*probab)
    return (entrpy)

#print (entropy(train_arr_data))




def node_entropy(split1, split2):
    p1 = len(split1)/(len(split1) + len(split2))
    p2 = len(split2)/(len(split1) + len(split2))

    entrpy = p1*entropy(split1) + p2*entropy(split2)
    return entrpy

def node_entropy_gini(split1, split2):
    p1 = len(split1)/(len(split1) + len(split2))
    p2 = len(split2)/(len(split1) + len(split2))

    entrpy = p1*entropy_gini(split1) + p2*entropy_gini(split2)
    return entrpy

def gain_ratio(arr_data, split1, split2):
    split_entrpy = node_entropy(split1, split2)
    gain = split_entrpy - entropy(arr_data)
    ratio = gain/split_entrpy
    return ratio
     

#s1,s2 = splits(train_df[0:100], 2, 205019)
#print (node_entropy(s1,s2))

def median_string(v):
    a = np.sort(v)
    if (len(a)%2 == 0):
        median = v[(len(a)/2-1)]
    else:
        median = v[(len(a)-1)/2]
    return median

def best_split_thresh(data):
    arr_data = data.values
    possible = splits_poss(arr_data)
    temp_entpy = 1000000
    for i in range(len(data.columns)-1):
       # print (len(possible[i]))
        for j in possible[i]:
            #print(j)
            s1,s2 = splits(data, i, j)
            entrpy = node_entropy(s1,s2)
            if temp_entpy >= entrpy:
                temp_entpy = entrpy
                best_attr = i
                best_val = j
    return best_attr, best_val


def best_split_median(data):
    arr_data = data.values
    types = feature_type(data)
    colm = data.columns
    count = 0
    temp_entpy = 1000000
    for i in colm:
        if i != 'Rich?' and count != 14:
            col = data[i]
            if (types[count] == "continous"):
                split_val = np.median(col)
                s1,s2 = splits(data, count, split_val)
                entrpy = node_entropy(s1,s2)
                if (entrpy <= temp_entpy):
                    temp_entpy = entrpy
                    best_colm = count
                    best_split = split_val
            else:
                split_val = median_string(col)
                s1,s2 = splits(data, count, split_val)
                entrpy = node_entropy(s1,s2)
                if (entrpy <= temp_entpy):
                    temp_entpy = entrpy
                    best_colm = count
                    best_split = split_val
            count += 1
    
    return best_colm, best_split


def best_split_median_gratio(data):
    arr_data = data.values
    types = feature_type(data)
    colm = data.columns
    count = 0
    temp_gain = -1111111
    for i in colm:
        if i != 'Rich?' and count != 14:
            col = data[i]
            if (types[count] == "continous"):
                split_val = np.median(col)
                s1,s2 = splits(data, count, split_val)
                gain = gain_ratio(arr_data,s1,s2)
                if (gain >= temp_gain):
                    temp_gain = gain
                    best_colm = count
                    best_split = split_val
            else:
                split_val = median_string(col)
                s1,s2 = splits(data, count, split_val)
                gain = gain_ratio(arr_data,s1,s2)
                if (gain >= temp_gain):
                    temp_gain = gain
                    best_colm = count
                    best_split = split_val
            count += 1
    
    return best_colm, best_split


def best_split_median_gini(data):
    arr_data = data.values
    types = feature_type(data)
    colm = data.columns
    count = 0
    temp_entpy = 1000000
    for i in colm:
        if i != 'Rich?' and count != 14:
            col = data[i]
            if (types[count] == "continous"):
                split_val = np.median(col)
                s1,s2 = splits(data, count, split_val)
                entrpy = node_entropy_gini(s1,s2)
                if (entrpy <= temp_entpy):
                    temp_entpy = entrpy
                    best_colm = count
                    best_split = split_val
            else:
                split_val = median_string(col)
                s1,s2 = splits(data, count, split_val)
                entrpy = node_entropy_gini(s1,s2)
                if (entrpy <= temp_entpy):
                    temp_entpy = entrpy
                    best_colm = count
                    best_split = split_val
            count += 1
    
    return best_colm, best_split



def convert_to_data(arr_data, features):
    data = pd.DataFrame(columns = features, data  = arr_data)
    return data







class Node:
    def __init__(self,data):
        self.left = None
        self.right = None
        self.quest = None
        self.val = data
        self.level = None



root = Node(train_df) 
def decision_tree(node, max_depth, min_size, count, crit):
    #print (count)
    data = node.val
    arr_data = data.values
    features = data.columns
    if ((is_same(arr_data))  or count >= max_depth or len(arr_data) <= min_size):
        #print("hi")
      #  classy = classification(arr_data)
        node.level = count
        return node
        #print(classy)
       # return classy
        
    else:
        count = count + 1
        if crit == "gini":
            split_column, split_val = best_split_median_gini(data)
        elif crit == "IG":
            split_column, split_val = best_split_median(data)
        elif crit == "GR":
            split_column, split_val = best_split_median_gratio(data)
       # print(split_column, split_val)
        split1, split2 = splits(data, split_column, split_val)
        #print(len(split1),len(split2))
        if (len(split1) ==0 or len(split2) == 0):
            #print(classification(arr_data))
            #node.val = data
            node.level = count
            return node
           # return(classification(arr_data))
        
        if feature_type(data)[split_column] == "categorical":
           # print("yes",count)
            node.quest = str(features[split_column]) + "==" + str(split_val)
        else:
            node.quest = str(features[split_column]) + "<=" + str(split_val)
            
        left_ = Node(convert_to_data(split1, features))
        right_ = Node(convert_to_data(split2, features))
        node.level = count
        node.left = decision_tree(left_,max_depth, min_size, count, crit)
        node.right = decision_tree( right_,max_depth, min_size, count, crit)
        return (node)

 #VISUALIZATION STARTS         
def BFT(node):

    node.level = 1
    queue = deque([node])
    output = []
    current_level = node.level

    while len(queue)>0:

          current_node = queue.popleft()

          if(current_node.level > current_level):
              output.append("\n")
              current_level += 1

          output.append(str(current_node.quest))

          if current_node.left != None:
             current_node.left.level = current_level + 1 
             queue.append(current_node.left) 

          if current_node.right != None:
             current_node.right.level = current_level + 1 
             queue.append(current_node.right)

    print(''.join(output))
         
 #VISUALIZATION ENDS




def classify_new(eg, tree, data= train_df):
    features = data.columns
    #print(features)
   # arr_data = data.values
    question = tree.quest
   # print(question)
    if tree.left == None and tree.right == None:
      #  print(classification(tree.val.values))
        return classification(tree.val.values)
    else:
        op = question.find("==") 
        if op != -1:
            feature = question[:op]
            #print(op)
            
            #print(feature)
            col_index = pd.Index.get_loc(features, feature)
            split_val = str(question[op+2:len(question)])
            if (eg[col_index] == split_val):
                if (tree.left == None):
                 #   print(classification(tree.val.values))
                    return classification(tree.val.values)
                else:
                    return classify_new(eg, tree.left)
            else:
                if (tree.right == None):
                 #   print(classification(tree.val.values))
                    return classification(tree.val.values)
                else:
                    return classify_new(eg, tree.right)
        else:
            op = question.find("<=")
            feature = question[:op]
            #print(op)
            
            #print(feature)
            col_index = pd.Index.get_loc(features, feature)
            #print(question[op+2:len(question)])
            split_val = float(question[op+2:len(question)])
            
            if (eg[col_index] <= split_val):
                if (tree.left == None):
                    #print(classification(tree.val.values))
                    return classification(tree.val.values)
                else:
                    return classify_new(eg, tree.left)
            else:
                if (tree.right == None):
                   # print(classification(tree.val.values))
                   return classification(tree.val.values)
                else:
                    return classify_new(eg, tree.right)
    
        
            
def accuracy(data, tree):
    arr_data = data.values
    #print(arr_data)
    #print(len(arr_data[0]))
    feature1 = arr_data[:, :len(arr_data[0])-1]
    label = arr_data[:, len(arr_data[0])-1:len(arr_data[0])]
    #print(label)
    #print(feature1.shape)
    prediction = np.zeros((len(arr_data),1))
    #print(prediction.shape)
    for i in range(len(feature1)):
        prediction[i] = classify_new(feature1[i], tree_root)
    correct = (label == prediction)
    acc = correct.mean()
    return acc


def pruning(tree,old_accur = 0):
    if tree.left == None:
        if tree.right == None:
            accur = accuracy(valid_df, tree)
            return accur
        else:
            old_accur = accuracy(valid_df, tree)
            right_ = tree.right
            tree.right = None
            accur = accuracy(valid_df, tree)
            if old_accur < accur:
                old_accur = accur
                return accur
            else:
                tree.right = right_
                return pruning(tree.right)
    else:
        if tree.left == None:
            accur = accuracy(valid_df, tree)
            return accur
        else:
            old_accur = accuracy(valid_df, tree)
            left_ = tree.left
            tree.left = None
            accur = accuracy(valid_df, tree)
            if old_accur < accur:
                old_accur = accur
                return accur
            else:
                tree.left = left_
                return pruning(tree.left)
    

#BFT(tree_root)
#print(accuracy(valid_df, tree_root))
#print(classify_new(train_arr_data[1], tree_root))
#print(train_arr_data[1][14])
#print(tree_root.val)
#print((tree_root).quest)
#print((tree_root.left).quest)
#print((tree_root.right).quest)

#print(decision_tree(v1))
accuracies = []
nodes =[]
for i in range (1,11):
    v1 = Node(train_df)
    tree_root = v1
    print(i)
    tree_root = decision_tree(v1,i,0,0,"gini")
    print(pruning(tree_root))
    accuracies.append(pruning(tree_root))
    nodes.append(pow(2,i))
print("gini")
print(nodes)
print(accuracies)




accuracies = []
nodes =[]
for i in range (1,11):
    v1 = Node(train_df)
    tree_root = v1
    print(i)
    tree_root = decision_tree(v1,i,0,0,"IG")
    print(pruning(tree_root))
    accuracies.append(pruning(tree_root))
    nodes.append(pow(2,i))
print("IG")
print(nodes)
print(accuracies)




accuracies = []
nodes =[]
for i in range (1,11):
    v1 = Node(train_df)
    tree_root = v1
    print(i)
    tree_root = decision_tree(v1,i,0,0,"GR")
    print(pruning(tree_root))
    accuracies.append(pruning(tree_root))
    nodes.append(pow(2,i))
print("GR")
print(nodes)
print(accuracies)


'''
v1 = Node(train_df)
tree_root = decision_tree(v1,1,50,0,"IG")
print(pruning(tree_root))
''' 

    
    


