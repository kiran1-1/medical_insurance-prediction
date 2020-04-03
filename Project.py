import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
data=pd.read_csv("insurance_dataset.csv")
print(data.info)



# Tried to impute the data to find and replace any missing value but faced error as smoker and sex columns are not float value

data.smoker[data.smoker == 'yes'] = 0
data.smoker[data.smoker == 'no'] = 1

data.sex[data.sex == 'female'] = 0
data.sex[data.sex == 'male'] = 1

data.region[data.region == 'southeast'] = 1
data.region[data.region == 'southwest'] = 2
data.region[data.region == 'northeast'] = 3
data.region[data.region == 'northwest'] = 4
data["obesity"] = ""
data.obesity[data.bmi >=30] = 1
data.obesity[data.bmi <30] = 0
data=data[['age','sex','bmi','children','smoker','region','obesity','charges']]
print(data.values.T)

x=data.iloc[:,:-1].values
y=data['charges'].values
print(x.shape)
print(y.shape)
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
x = imputer.fit_transform(x)
y = y.reshape(-1,1)
y = imputer.fit_transform(y)
y = y.reshape(-1)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
data_scaled = scaler.fit_transform(data)
print(data_scaled.mean(axis=0))
print(data_scaled.std(axis=0))

print(x.shape)
print(y.shape)
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp

corr_vals = []
collabel = []
for col in [i for i in data.columns]:
    p_val = sp.stats.pearsonr(data[col], data["charges"])
    corr_vals.append(np.abs(p_val[0]))
    print(col, ": ", np.abs(p_val[0]))
    collabel.append(col)
plt.bar(range(1, len(corr_vals) + 1), corr_vals)
plt.xticks(range(1, len(corr_vals) + 1), collabel, rotation=45)
plt.ylabel("Absolute correlation")
plt.show()



#Predicting Value using Linear Regression
print("Linear Regression")
from sklearn.model_selection import train_test_split
(x_train,x_test,y_train,y_test)=train_test_split(x,y,test_size=0.3,random_state=0)
print('Training Data ',x_train.shape,y_train.shape)
print('Testing Data ',x_test.shape)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_test_pred=lr.predict(x_test)
y_train_pred=lr.predict(x_train)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print('MSE-> Train: %.3f,Test:%.3f'%(mean_squared_error(y_train,y_train_pred), mean_squared_error(y_test,y_test_pred)))
print('R2_Score ->  Train: %.3f,Test:%.3f'%(r2_score(y_train,y_train_pred), r2_score(y_test,y_test_pred)))

#Predicting Value Using Random Forest Regression

print("Random Forest Regression")

from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor(n_estimators=1000, criterion='mse',random_state=1, n_jobs=-1)
forest.fit(x_train,y_train)
y_train_pred=forest.predict(x_train)
y_test_pred=forest.predict(x_test)
print(forest.predict(x_train[:1]))
print(x_train[:1])
# Computing  Performance

from sklearn.metrics import mean_squared_error
t_train=mean_squared_error(y_train,y_train_pred)
t_test=mean_squared_error(y_test,y_test_pred)
print('MSE -> Train: %.3f, MSE test: %.3f'%(t_train,t_test))
from sklearn.metrics import r2_score
t_train1=r2_score(y_train,y_train_pred)
t_test1=r2_score(y_test,y_test_pred)
print('R2 Score-> Train: %.3f, MSE test: %.3f'%(t_train1,t_test1))
