# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn

## Program:
```

Developed by: HARSSHITHA LAKSHMANAN
RegisterNumber: 212223230075

import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![decision tree classifier model](sam.png)
![image](https://github.com/harshulaxman/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145686689/5a9e8613-caf1-43fe-9ceb-76a3f6343bec)
![image](https://github.com/harshulaxman/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145686689/c46d8cb2-5c6b-4f6a-a952-2918112401b6)
![image](https://github.com/harshulaxman/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145686689/d43016c4-78af-473b-a03b-2d1174233fec)
![image](https://github.com/harshulaxman/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145686689/a0d5ba62-bf62-496f-b386-e511a5c33706)
![image](https://github.com/harshulaxman/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145686689/c09d06b7-9c7c-401b-a462-d09ec5e747ff)
![image](https://github.com/harshulaxman/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145686689/015c1bc9-bff4-489e-ab5c-ce6ebefcf953)
![image](https://github.com/harshulaxman/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145686689/5265c773-ae3c-4f42-b8a0-c0e97fe932dc)
![image](https://github.com/harshulaxman/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/145686689/8cde484a-9981-4b17-b2e9-9e1adf65cb80)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
