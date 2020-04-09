import tkinter as tk


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





# tkinter GUI
root = tk.Tk()

canvas1 = tk.Canvas(root, width=500, height=400)
canvas1.pack()


# New_Interest_Rate label and input box
label1 = tk.Label(root, text='Age : ')
canvas1.create_window(120, 100, window=label1)

entry1 = tk.Entry(root)  # create 1st entry box
canvas1.create_window(270, 100, window=entry1)


label2 = tk.Label(root, text=' Gender: ')
canvas1.create_window(120, 120, window=label2)

#entry2 = tk.Entry(root)  # create 2nd entry box

entry2 = tk.Radiobutton(root,text='Male',padx=20,value=0).place(x=200,y=110)
entry2 = tk.Radiobutton(root,text='Female',padx=20,value=1).place(x=300,y=110)


#canvas1.create_window(270, 120, window=entry2)

label3 = tk.Label(root, text=' Weight in kg: ')
canvas1.create_window(120, 140, window=label3)

entry3 = tk.Entry(root)  # create 2nd entry box
canvas1.create_window(270, 140, window=entry3)


label4 = tk.Label(root, text=' Height in cm: ')
canvas1.create_window(120, 160, window=label4)

entry4 = tk.Entry(root)  # create 2nd entry box
canvas1.create_window(270, 160, window=entry4)


label5 = tk.Label(root, text=' Children: ')
canvas1.create_window(120, 180, window=label5)

entry5 = tk.Entry(root)  # create 2nd entry box
canvas1.create_window(270, 180, window=entry5)


label6 = tk.Label(root, text=' Smoker: ')
canvas1.create_window(120, 200, window=label6)

entry6 = tk.Entry(root)  # create 2nd entry box
canvas1.create_window(270, 200, window=entry6)

label7 = tk.Label(root, text=' Region: ')
canvas1.create_window(120, 220, window=label7)

#entry7 = tk.Entry(root)  # create 2nd entry box
entry7 = tk.Radiobutton(root,text='SouthEast',padx=20,value=1).place(x=270,y=215)
entry7 = tk.Radiobutton(root,text='SouthWest',padx=20,value=2).place(x=270,y=235)
entry7 = tk.Radiobutton(root,text='NorthEast',padx=20,value=3).place(x=270,y=255)
entry7 = tk.Radiobutton(root,text='NorthWest',padx=20,value=4).place(x=270,y=275)
#canvas1.create_window(270, 220, window=entry7)


def values():
    global age_value  # our 1st input variable
    age_value = float(entry1.get())

    sex_value = 0  # our 2nd input variable
    if entry2 == 0:
        sex_value = 0.0
    elif entry2 == 1:
        sex_value = 1.0

    global bmi_value  # our 2nd input variable
    weight = float(entry3.get())
    height = float(entry4.get())
    if height==0:
        height=1
    bmi_value=weight/(height/100)

    global children_value  # our 2nd input variable
    children_value = float(entry5.get())

    global smoker_value  # our 2nd input variable
    smoker = entry6.get()
    if smoker == "yes":
        smoker_value = 0
    else :
        smoker_value = 1
    smoker_value = float(smoker_value)


    region_value = 0
    if entry7 == 1:
        region_value = 1.0
    elif entry7 == 2:
        region_value = 2.0
    elif entry7 == 3:
        region_value = 3.0
    elif entry7 == 4:
        region_value = 4.0



    global obesity_value
    if bmi_value >= 30:
        obesity_value = 1.0
    else:
        obesity_value = 0.0

    ready_value = [age_value, sex_value, bmi_value, children_value, smoker_value, region_value, obesity_value]

    pr1 = lr.predict([ready_value])
    pr2 = forest.predict([ready_value])
    pr = (pr1+pr2)/2
    pr = abs(pr)

    Prediction_result = ('Predicted Cost of Insurance: ', pr)
    label_Prediction = tk.Label(root, text=Prediction_result,font = "Helvetica 9 bold italic",fg='light green', bg='teal')
    canvas1.create_window(320, 350, window=label_Prediction)


button1 = tk.Button(root, text='Predict Insurance Cost', command=values,
                    bg='teal')  # button to call the 'values' command above
canvas1.create_window(100, 350, window=button1)
label_main = tk.Label(root,text="Insurance Cost Prediction Tool",fg='light green',font = "Helvetica 16 bold italic",bg='teal')
canvas1.create_window(250,35,window=label_main)

root.mainloop()
