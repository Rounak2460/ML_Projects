
Project1 : Regression Models for multi-class prediction on Blog Feedback Dataset and Nursery Dataset
    
    a: implementation of  Moore-Penrose pseudo inverse 
    weightfile.txt will be destiantion to store output weights 
    Run file in this form : python 1a.py a trainﬁle.csv testﬁle.csv outputﬁle.txt weightﬁle.txt
    Dataset Link: https://archive.ics.uci.edu/ml/datasets/BlogFeedback
    
    b: Ridge regression implementation
    here regularization has set of regularization parameter values
    run file as: python linear.py b trainﬁle.csv testﬁle.csv regularization.txt outputﬁle.txt               weightﬁle.txt
    
    c. Feature engineered version with ridge regression
    run file as: python linear.py c trainﬁle.csv testﬁle.csv outputﬁle.txt 
    
    1d. Implementation of logistic regression:
     Implementation using k class logistic regression with batch grad descent
     param.txt will contain three lines of input, the ﬁrst being a number [1-3] indicating which          learning rate strategy to use and the second being the ﬁxed learning rate (for ”1”), seed value      for adaptive learning rate (for ”2”) or a comma seperated (learning rate, α, β) value for αβ          backtracking. The third line will be the max number of iterations. Also print the number of          iterations your program takes.
    
    run file as: python logistic a.py trainﬁle.csv testﬁle.csv param.txt outputﬁle.csv weightﬁle.csv 
    
    1e. Implementation of minibatch grad descent iver nursey dataset
    run as: python logistic b.py trainﬁle.csv testﬁle.csv param.txt outputﬁle.csv weightﬁle.csv 
   
    Nursery data set: https://archive.ics.uci.edu/ml/datasets/nursery 

Project 2: Neural Network for Binary and Multi class Image Classiﬁcation

      Param.txt will contain ﬁve lines of input, the ﬁrst being a number [1-2] indicating which learning rate strategy to use and the second being the ﬁxed learning rate (for ”1”), seed value for adaptive learning rate (for ”2”). The third line will be the max number of iterations. The fourth line will contain batch size for one iteration of mini batch gradient descent. Fifth line will contain the architecture of the neural network with a sequence of numbers denoting number of perceptrons in consequent layers eg. 10 10 5 denotes 3 hidden layers with 10, 10 and 5 perceptrons respectively. 

       2a. Manual implementation of neural netwroks for Toy Dataset:
       run as: python neural a.py trainﬁle.csv param.txt weightﬁle.txt 
       2b: Manual Implementation for CIFAR-10 Dataset
       python neural b.py trainﬁle.csv param.txt weightﬁle.txt 
       
    Toy: https://www.kaggle.com/carlolepelaars/toy-dataset
    CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
  
Project 3: Image classification for CIFAR-10 and CIFAR-100 Dataset
  
          Keras neural network architecture for the two datasets.
          Applying new and modified SOTA for CIFAR 100 and CIFAR 10
          Files will be updated till december to achieve higher accuracy every time with different             architectures.
          run file as: python compete.py trainﬁle.csv testﬁle.csv output.txt 
          CIFAR 10 & CIFAR-100 Datset: https://www.cs.toronto.edu/~kriz/cifar.html
  
Project 4: Decision Tree for Adult Dataset Classiﬁcation
            
         A manual implementation of a pruned decision tree with various splitting methods.
         Input needs a train and a valid file thus datset is to be splitted by user. 
         run as: python dt.py trainﬁle.csv validﬁle.csv testﬁle.csv validpred.txt testpred.txt 
         Datset from : https://archive.ics.uci.edu/ml/datasets/adult
       
       

    
