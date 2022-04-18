# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.

2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

4. Import LogisticRegression from sklearn and apply the model on the dataset.

5. Predict the values of array.

6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7. Apply new unknown values.

## Program:
```
'''
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SRIJITH R
RegisterNumber:  212221240054
'''

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])

data1

x=diata1.iloc[:,:-1]
x

y=data1["status"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```

### Output:

## HEAD OF THE DATA:

![OUTPUT1](https://user-images.githubusercontent.com/93427240/162576788-989e90ee-8127-4058-b543-f5994dfdcbb8.png)


## PREDICTED VALUES:

![OUTPUT2](https://user-images.githubusercontent.com/93427240/162576818-7d3d2e64-b087-4e8c-b57a-f37686429d84.png)


## ACCURACY:

![OUTPUT3](https://user-images.githubusercontent.com/93427240/162576829-34bbe153-1466-45ac-9086-c759e4eec41f.png)


## CONFUSION MATRIX:

![OUTPUT4](https://user-images.githubusercontent.com/93427240/162576861-86a14312-aacb-4d36-85a3-aeca60f56054.png)


## CLASSIFICATION REPORT:

![OUTPUT5](https://user-images.githubusercontent.com/93427240/162576889-7346bc85-4ecc-41b5-aac4-c6f8df109e53.png)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
