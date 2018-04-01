import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd
#some math in svm:
#1. decision rule: w • u + b >=0 then positive sample
#2. constraints: yi(xi * w + b) -1 = 0
#3. maximize the width -> minimize \w\

# pros for svm : good at dealing with high dimensional data; works well on small data sets
# cons: picking the right kernel and parameters

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace=True)# deal with missing data
df.drop(['id'],1,inplace=True)#remove the id column

x = np.array(df.drop(['class'],1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2)

clf = svm.SVC()
clf.fit(x_train,y_train)

accuracy = clf.score(x_test,y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures),-1)
predict_val = clf.predict(example_measures)
print(predict_val)

