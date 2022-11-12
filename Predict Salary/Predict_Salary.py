from pyexpat import model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('C:\\Users\\Sabari\\OneDrive\\Desktop\\New folder (9)\\Predict Salary\\Salary_Data.csv')
print(data.to_string())
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(data[['YearsExperience']],data[['Salary']])
print(model.predict([[14]]))

