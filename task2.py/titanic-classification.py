import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math 
file_path1 = "test.csv"
file_path2 = "train.csv"
test = pd.read_csv(file_path1)
train = pd.read_csv(file_path2)
# train.head()
# test.head()
# train.shape
sns.countplot(x='Survived',data=train)
# train
sns.countplot(x='Survived',hue='Sex',data=train ,palette='winter')
sns.countplot(x='Survived',hue='Pclass', data=train, palette='PuBu')
train['Age'].plot.hist()

train['Fare'].plot.hist(bins=20,figsize=(10,5))

sns.countplot(x='SibSp',data=train,palette='rocket',legend=False)
train['Parch'].plot.hist()

sns.countplot(x='Parch',data=train,palette='summer')
# train.isnull().sum()

sns.heatmap(train.isnull(),cmap='spring')
sns.boxplot(x='Pclass',y='Age',data=train)
train.drop('Cabin',axis=1,inplace=True)
# train.head()
train.dropna(inplace=True)
sns.heatmap(train.isnull(),cbar=False)

# train.isnull().sum()
pd.get_dummies(train['Sex']).head()
sex=pd.get_dummies(train['Sex'],drop_first=True)
# sex.head(3)
embark=pd.get_dummies(train['Embarked'])
# embark.head()
embark=pd.get_dummies(train['Embarked'],drop_first=True)
Pcl=pd.get_dummies(train['Pclass'],drop_first=True)
# Pcl.head(3)
titanic=pd.concat([train,sex,embark,Pcl],axis=1)
# train.head()
train.drop(['Name','PassengerId','Pclass','Ticket','Sex','Embarked'],axis=1,inplace=True)

# train.head()
X= train.drop('Survived',axis=1)
y=train['Survived']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
lm=LogisticRegression()

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.33,random_state=4)
lm.fit(X_train, y_train)

X_test.columns = X_test.columns.astype(str)
prediction=lm.predict(X_test)
from sklearn.metrics import confusion_matrix 

confusion_matrix(y_test,prediction)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction)