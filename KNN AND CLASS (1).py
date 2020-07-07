#!/usr/bin/env python
# coding: utf-8

# KNN IN JUST 3 STEPS
# 1. CALCULATE DISTANCE.
# 2. GET NN
# 3. MAKE PREDICTION.

# In[1]:


import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
df = pd.read_csv("iris.csv")
df


# In[2]:


df.variety =  df.variety.map({
    "Setosa":0,
    "Versicolor":1,
    "Virginica":2
    
})

irisdata = df.values
sfl = np.random.permutation(len(irisdata))
irisdata = irisdata[sfl]

training_data = irisdata[:10]


# In[3]:


## Find Euclidean Distance
def Euclidean_Distance(row1, row2):
    distance = 0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return distance**0.5

new_data = [4.2, 2.9, 1.0, 0.7, 1]

all_dis = []
for row in training_data:
    out = Euclidean_Distance(new_data, row)
    all_dis.append(out)
    

def Get_NN(train, test_row, k = 3):
    distance = []
    data = []
    for i in train:
        dist = Euclidean_Distance(test_row, i)
        distance.append(dist)
        data.append(i)
    ## convert distance and data in nump array
    distance = np.array(distance)
    data = np.array(data)
    "*** we are finding index number of min distance  ***"
    index_dist = distance.argsort()
    "** arange Data acco. to index_dist **"
    data = data[index_dist]
    "** find the nn with slicing on data acco. to K value  **"
    neighbors = data[:k]
    return neighbors
    

    
    


# In[4]:


all_nn = Get_NN(training_data, new_data, 10)
all_nn


# In[5]:


def Prediction_Classes(train,test_row, k):
    Neighbors = Get_NN(train, test_row, k)
    classes = []
    for i in Neighbors:
        classes.append(i[-1])
    prediction = max(classes, key = classes.count)
    return prediction


# In[6]:


prediction = Prediction_Classes(irisdata, irisdata[12],10)
prediction


# In[7]:


def Predict(Train, test, k=3):
    y_predict = []
    for i in test:
        prediction = Prediction_Classes(Train,i,k)
        y_predict.append(prediction)
    return np.array(y_predict)

def Evaluate(y_true, y_pred):
    n_correct = 0
    for i,j in zip(y_true, y_pred):
        if i == j:
            n_correct += 1
    return n_correct/len(y_true)


# In[9]:


y_pred = Predict(irisdata, irisdata, 6)
y_pred


# In[11]:


y_true = irisdata[:,-1]
y_true


# In[12]:


Evaluate(y_true, y_pred)


# ALL THE FUNCTION IN SINGLE CLASS.
# 

# 2)HAND WRITTEN DIGIT CLASSIFICATION.

# In[16]:


from sklearn.datasets import fetch_openml #openml is used to fetch the data from the browser.
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[17]:


digits = fetch_openml("mnist_784")
digits.target = digits.target.astype(np.int8) #convering into integer
digits.keys()


# In[18]:


# here we need only data and target.
x = digits.data
y = digits.target
digits.feature_names # finding classes #784 because they split our imgage into 28*28


# In[19]:


x.shape, y.shape


# In[20]:


some_digits = x[12]
some_digits_image = some_digits.reshape(28,28)
plt.imshow(some_digits_image, cmap=matplotlib.cm.binary) ##here cmap is used to convert coloured image of 3 into handwritten image
plt.axis("off") #removing axis
plt.show()


# In[21]:


trainx, testx, trainy, testy = train_test_split(x,y, test_size = 0.3, random_state = 12)


# In[22]:


class KNN: # calling all the function in one lib.
    n_neighbors = 3 #global variable
    train_x = []
    train_y = [] #storing all the data of y
    pred_y = []
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors #above(3) n_neighbors will replace by below one
    def fit(self,x,y):
        if np.ndim(x) != 2:
            raise ValueError("Dim of training data shoul be 2D,Got:" + str(np.ndim(x)))
        if len(x) != len(y):
            raise ValueError("Len of x an d y shoul be same, got. {}, {}".format(x.sahpe,y.shape))
        self.train_y = y
        self.train_x = x
        return self    
            
            
    # Find Euclidean Distance
    def Euclidean_Distance(self,row1, row2):
        distance = 0
        for i in range(len(row1)-1):
            distance += (row1[i] - row2[i])**2
        return distance**0.5


    def Get_NN(self,train, test_row, k = 3):
        distance = []
        data = []
        for i in train:
            dist = self.Euclidean_Distance(test_row, i)
            distance.append(dist)
            data.append(i)
        ## conver distance and data in nump array
        distance = np.array(distance)
        data = np.array(data)
        "*** we are finding index number of min distance  ***"
        index_dist = distance.argsort()
        "** arange Data acco. to index_dist **"
        data = data[index_dist]
        "** find the nn with slicing on data acco. to K value  **"
        neighbors = data[:k]
        return neighbors


    def Predict_Classes(self,train, test_row, k):
        Neighbors = self.Get_NN(train, test_row, k)
        classes = []
        for i in Neighbors:
            classes.append(i[-1])
        prediction = max(classes, key = classes.count)
        return prediction


    def Predict(self,test):
        y_predict = []
        n = 0
        Train = np.insert(self.train_x, len(self.train_x[0]), self.train_y, axis=1) # merge train_x and train_y, in last columns(x[0] )
        
        for i in test:
            i[-1] = 1
            prediction = self.Predict_Classes(Train, i, self.n_neighbors)
            y_predict.append(prediction)
            n = n+1
            print("steps-" + str(n))
        self.pred_y = np.array(y_predict)
        return np.array(y_predict)


    def Evaluate(self,y_true):
        n_correct = 0
        for i,j in zip(y_true, self.pred_y):
            if i == j:
                n_correct += 1
        return n_correct/len(y_true)
    
   


# In[23]:


knn_class = KNN(n_neighbors=6)
knn_class = knn_class.fit(trainx[:50],trainy[:50])
knn_class.Predict(trainx[:50])


# In[24]:


knn_class.Evaluate(trainy[:50])


# In[25]:


## WITH THE HELP OF LIB


# In[26]:


import numpy as np
import pandas as pd
df = pd.read_csv("iris.csv")


# In[27]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
x = irisdata[:,:-1]
y = irisdata[:,-1]
KNN_Class = KNeighborsClassifier(n_neighbors=6)
KNN_Class = KNN_Class.fit(x,y)


# In[28]:


y_pred_sklearn = KNN_Class.predict(x)


# In[29]:


y_pred_sklearn


# In[30]:


Evaluate(y,y_pred_sklearn)


# In[31]:


accuracy_score(y,y_pred_sklearn)


# In[32]:


confusion_matrix(y,y_pred_sklearn)


# In[ ]:





# 

# In[12]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




