from sklearn.linear_model import LinearRegression
import pandas as pd

# 'YearsExperience', 'Salary'

path = r"D:\AI\instant program\Machine Learning\Regresion\One Variable\Tasks\Data\Salary_dataset.csv"
Salary = pd.read_csv(path)
Salary.drop("Unnamed: 0" , axis = 1, inplace = True)
x = Salary[['YearsExperience']] # return one Dinmension to 2 Dimension
y = Salary[['Salary']]
model = LinearRegression()
model.fit(x, y)
print(  "the Slope is  :- " ,model.coef_[0])
print( "the Intercept is  :- " ,model.intercept_)
print(model.predict([[10]]))
print(model.score(x, y))
