#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import re, string
import decimal
from collections import Counter

with open('train.dat') as f:
    train = f.read()
with open('train.labels') as f:
    labels = f.read()
with open('test.dat') as f:
    test = f.read()

train = train.lower()
regex = re.compile('[%s]' % re.escape(string.punctuation))     
train=regex.sub('', train) 
test = test.lower()
regex = re.compile('[%s]' % re.escape(string.punctuation))     
test=regex.sub('', test) 

train_split = train.split('\n')
filtered_text = ' '.join(train_split)
filtered_text = filtered_text.split()
test_split = test.split('\n')
filtered_test = ' '.join(test_split)
filtered_test = filtered_test.split()

train_int = {}
test_int = {}
i = 1
for x, y in Counter(filtered_test).most_common():
        test_int[x] = i
        i+=1
for x, y in Counter(filtered_text).most_common():
    train_int[x] = i
    i+=1


train_index = []
for sent in train_split:
    a = []
    for word in sent.split():
        a.append(train_int[word])
    train_index.append(a)
test_index = []
for s in test_split:
    a = []
    for word in s.split():
        a.append(test_int[word])       
    test_index.append(a)

labels=labels.split('\n')

train_lengths = Counter([len(x) for x in train_index])
test_lengths = Counter([len(x) for x in test_index])

train_idx = [i for i, ele in enumerate(train_index) if len(ele)!=0]
train_index = [train_index[i] for i in train_idx]
labels = np.array([labels[i] for i in train_idx])


test_idx = [i for i, ele in enumerate(test_index) if len(ele)!=0]
test_index = [test_index[i] for i in test_idx]


def slicing_train(train_index, cols):
    features_train = []
    
    for rev in train_index:
        if len(rev) >= seq_length:
            features_train.append(rev[:cols])
        else:
            features_train.append([0]*(cols-len(rev)) + rev)
    
    return np.array(features_train)


def slicing_test(test_index, cols):
    features_test = []
    
    for rev in test_index:
        if len(rev) >= seq_length:
            features_test.append(rev[:cols])
        else:
            features_test.append([0]*(cols-len(rev)) + rev)
    
    return np.array(features_test)


cols = 2000

features_train = slicing_train(train_index, cols)
features_test = slicing_test(test_index, cols)

train_x = np.array(features_train)
test_x = np.array(features_test)
train_y = np.array(labels).astype(np.int64)
#train_x.shape
#train_y.shape
#type(test_x.shape)

def cost_func(X,y,theta):
         z=np.dot(X, theta)
         np.log(0)
         return np.dot(y, np.log(z)) + np.dot(1 - y, np.log(1-z))

def sigmoid(x):
         return 1.0/(1.0+(np.exp(-x)))

def predict(X,w):
        y_predicted_cls=[]
        z=np.dot(X, w)
        y_test=sigmoid(z)
        for i in y_test:
            if i>=0.5:
                y_predicted_cls.append(+1)     
            elif i<0.5:
                y_predicted_cls.append(-1)
        return y_predicted_cls  
    
def fit(X,y):
        theta = np.zeros(X.shape[1])
        i=0
        C=1.0
        cost_list=[C]
        theta_list=[theta]
        while i in range(10000) and C >(0.000000001):
            z=np.dot(X, theta)
            C=cost_func(X,y,theta)
            #cost_list.append(C)
            print('cost_function',C)
            g = -np.mean((1.0-sigmoid(z)) * y * X.T, axis=1)
            theta=theta+0.8*g
            theta_list.append(theta)
            i=i+1
        print("Final Cost : ",C)
        print("Final Parameters - {0},{1},{2}".format(theta[0],theta[1],theta[2]))
        return theta
        
true=fit(train_x,train_y)
predictions=predict(test_x,true)      
        
with open('loutput.txt', 'w') as f:
            f.write('\n'.join(str(element) for element in predictions))


# In[ ]:




