#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# In[2]:


iris_data = load_iris()
X = iris_data.data
y = iris_data.target


# In[3]:


X_train_list = []
X_test_list = []
y_train_list = []
y_test_list = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    X_train_list.append(X_train)
    X_test_list.append(X_test)
    y_train_list.append(y_train)
    y_test_list.append(y_test)


# In[4]:


param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': [0.12, 0.15] + list(np.logspace(-3, 3, 7))
}


# In[5]:


best_params_list = []
accuracy_list = []
convergence_list = []
for i in range(10):
    X_train_sample = X_train_list[i]
    X_test_sample = X_test_list[i]
    y_train_sample = y_train_list[i]
    y_test_sample = y_test_list[i]
    svm = SVC(max_iter=1000)
    grid_search = GridSearchCV(svm, param_grid, cv=5)
    grid_search.fit(X_train_sample, y_train_sample)
    best_params = grid_search.best_params_
    best_params_list.append(best_params)
    svm_best = SVC(**best_params, max_iter=1000)
    accuracy_iteration = []
    for iter in range(1, 1001):
        svm_best.fit(X_train_sample, y_train_sample)
        y_pred = svm_best.predict(X_test_sample)
        accuracy = accuracy_score(y_test_sample, y_pred)
        accuracy_iteration.append(accuracy)
    accuracy_list.append(accuracy_iteration)
    convergence_list.append(accuracy_iteration[-1])


# In[6]:


# Find the sample with maximum accuracy
max_acc_index = np.argmax(convergence_list)
best_params_max_acc = best_params_list[max_acc_index]
accuracy_max_acc = accuracy_list[max_acc_index]


# In[8]:


#Making the list of best accuracy of 10 samples
indexes = [0,1,2,3,4,5,6,7,8,9]
Accuracy = []
for i in indexes:
    Accuracy.append(accuracy_list[i][999])


# In[9]:


df_best_params = pd.DataFrame(best_params_list)
df_best_params.index.name = 'Sample'
df_best_params['Accuracy'] = Accuracy
df_best_params.columns = ['C', 'gamma', 'kernel','Best Accuracy']


# In[10]:


print("Table 1: Best Parameters")
print(df_best_params)


# In[11]:


# Create line plot of accuracy convergence for sample with maximum accuracy
plt.plot(range(1, 1001), accuracy_max_acc)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy Convergence (Sample {})'.format(max_acc_index + 1))
plt.show()


# In[ ]:




