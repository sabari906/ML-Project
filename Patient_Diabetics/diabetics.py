import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data=pd.read_csv(r'filedirectory.csv')
X=data.drop(columns='Outcome', axis=1)
y=data['Outcome']

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)   

model=LogisticRegression()
model.fit(X_train, y_train)


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)


sample_data = (7,107,74,0,0,29.6,0.254,31)

chng_arry= np.asarray(sample_data)

input_data= chng_arry.reshape(1,-1)

my_pred = model.predict(input_data)

print(my_pred)

if (my_pred[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')