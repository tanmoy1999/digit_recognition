#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from sklearn.tree import DecisionTreeClassifier
import os
import sys
# %%

os.chdir(r'F:\datasci\Machine Learning')
data = pd.read_csv("train.csv").as_matrix()

clf = DecisionTreeClassifier()
# %% training dataset
xtrain = data[0:21000,1:]
train_label = data[0:21000,0] 

clf.fit(xtrain,train_label)

# %% testing data

xtest = data[21000:,1:]
actual_label = data[21000:,0]

i=11
d = xtest[i]
d.shape = (28,28)
pt.imshow(d,cmap='gray')
print(clf.predict([xtest[i]]))
pt.show()

# %%

p = clf.predict(xtest)
count = 0
else_count = 0

for i in range(0,21000):
    if p[i]==actual_label[i]:
        count+=1  
    else:
        print(i)
        else_count+=1
        0
print("Actutal label: ",count)
print("Not accurate: ",else_count)
print("accuracy = ",(count/21000)*100)


# %%
