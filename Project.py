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
print(data.values.T)

x=data.iloc[:,:-1].values
y=data.iloc[:-1].values
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
#linear_regression(prediction,precision,recall_score,f1_score)
x=data.iloc[:,:-1].values
y=data.iloc[:-1].values
from sklearn.model_selection import train_test_split
(x_train,x_test,y_train,y_test)=train_test_split(x,y,test_size=0.3,random_state=0)
print('Training Data ',x_train.shape)
print('Testing Data ',x_test.shape)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_train_pred=lr.predict(x_train)
y_test_pred=lr.predict(x_test)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print('MSE train: %.3f,Test:%.3f'%(mean_squared_error(y_train,y_train_pred), mean_squared_error(y_test,y_test_pred)))
print('R2_Score train: %.3f,Test:%.3f'%(r2_score(y_train,y_train_pred), r2_score(y_test,y_test_pred)))
#average precision score
from sklearn.metrics import average_precision_score
average_precision_training = average_precision_score(y_train, y_train_pred)
average_precision_testing= average_precision_score(y_test, y_test_pred)

print('Average precision-recall score for training: {0:0.2f}'.format(
      average_precision_training))
print('Average precision-recall score for testing: {0:0.2f}'.format(
      average_precision_testing))
#Recall 
from sklearn.metrices import recall_score
recall_score_training = recall_score(y_train,y_train_pred)
recall_score_testing = recall_score(y_test,y_test_pred)
print('Recall for training: %f' %recall_score_training)
print('Recall for testing: %f' %recall_score_testing)
#f1score
from sklearn.metrics import f1_score
f1_score_training = f1_score(y_train,y_train_pred)
f1_score_testing = f1_score(y_test,y_test_pred)
print('f1 score of training: %f' %f1_score_training)
print('f1 score of testing:%f' %f1_score_testing)
    #svm

 from sklearn import svm
from sklearn.model_selection import train_test_split
x=data.iloc[:,:-1].values
y=data.iloc[:-1].values
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size = 0.30,random_state=0)
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print('MSE Test:%.3f'%( mean_squared_error(y_test,y_pred)))
print('R2_Score Test:%.3f'%( r2_score(y_test,y_pred)))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
# k_nerist_neighbours




