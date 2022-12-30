# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 15:42:01 2022

@author: Derya
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import warnings

#import data
data = pd.read_csv("C:/Users/Derya/Desktop/Yapay Zeka Proje Ödevi/diabetes.csv")
data

#.............................................................

#APPLIED MODELS:
#Logistic Regression
#KNN
#SVM

# Creat DataFrame
diabets_df = pd.DataFrame(data)

diabets_df.describe(include = "all")

#........................................................................................................................

#DATA VISUALIZATION & ANALYSIS

diabets_df.head()

diabets_df.shape

# Show detaile columns
diabets_df.info()

# How many missing value?
diabets_df.isna().sum()

# Show detaile data set
desc = diabets_df.describe().T
diabets_df2 = pd.DataFrame(index=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'], 
                           columns= ["count","mean","std","min",
                                     "25%","50%","75%","max"], data= desc )

f,ax = plt.subplots(figsize=(12,12))

sns.heatmap(diabets_df2, annot=True,cmap = "Blues", fmt= '.0f',
            ax=ax,linewidths = 5, cbar = False,
            annot_kws={"size": 16})

plt.xticks(size = 18)
plt.yticks(size = 12, rotation = 0)
plt.ylabel("Variables")
plt.title("Descriptive Statistics", size = 16)
plt.show()

    
## plot for numerical columns
Numerical = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
i = 0
while i <8:
    fig = plt.figure(figsize =[20,4])
    plt.subplot(1,2,1)   #(one row, two plots, first one)
    sns.boxplot(x =Numerical[i], data = diabets_df )
    i+=1
    if i==8:
        break
    plt.subplot(1,2,2)
    sns.boxplot(x =Numerical[i], data = diabets_df)
    i+=1
  
    plt.show()
    
    
#histogram
diabets_df.hist(bins=50,figsize=(20,15))
plt.show()


diabets_df3 = diabets_df.copy()

zero_col = ['Glucose','Insulin','SkinThickness','BloodPressure','BMI']
diabets_df3[zero_col] = diabets_df3[zero_col].replace(0, np.nan)

for col in ['Glucose','Insulin','SkinThickness']:
    median_col = np.median(diabets_df3[diabets_df3[col].notna()][col])
    diabets_df3[col] = diabets_df3[col].fillna(median_col)
for col in ['BMI','BloodPressure']:
    mean_col = np.mean(diabets_df3[diabets_df3[col].notna()][col])
    diabets_df3[col] = diabets_df3[col].fillna(mean_col)
    
#histogram
diabets_df3.hist(bins=50,figsize=(20,15))
plt.show()

diabets_df3.isna().sum()

#.........................................................

## Show correlation
fig, ax = plt.subplots(figsize = (20, 12)) #Size of plot
ax = sns.heatmap(diabets_df3.corr(),cmap='RdBu_r',cbar=True,annot=True,linewidths=0.5,ax=ax)
plt.show()

#.........................................................

#Best correlation for Glucose

#correlation
diabets_df3.corr()

#.........................................................

diabets_df3.corr()['Outcome'].sort_values(ascending=False) #Correlation for get information

#.........................................................

#### How does Glucose is affected by price
sns.jointplot(x='Glucose',y='Outcome',data=diabets_df3,color='red',kind='kde');

#.....................................................................................................................


#CREATING & TRAINING KNN MODEL

# Sellecting features
X = pd.DataFrame(diabets_df3, columns = ["Glucose","BloodPressure", "SkinThickness", "Insulin","BMI","DiabetesPedigreeFunction","Age"]).values
Y = diabets_df3.Outcome.values.reshape(-1,1)
X



# Splitting the data
X_train, X_test, Y_train, Y_test  =  train_test_split(X,Y, test_size = 0.3, random_state = 0)



#MODELLING

K = 3
CLF = KNeighborsClassifier(K)
CLF.fit(X_train,Y_train.ravel() )
Y_pred = CLF.predict(X_test)



# Select Best value for K
K = 20
Acc = np.zeros((K))
for i in range(1 , K+1):
    CLF = KNeighborsClassifier(n_neighbors = i)
    CLF.fit(X_train,Y_train.ravel())
    Y_pred = CLF.predict(X_test)
    Acc[i-1] = metrics.accuracy_score(Y_test, Y_pred)
Acc



# Show max & min
print(np.max(Acc))
print(np.min(Acc))

#....................................................................

########  Another way

traing_acc = []
test_acc = []
# try KNN for diffrent k nearset neighbor from 1 ta 50
neighbors_setting = range(1,20)

for n_neighbors in neighbors_setting:
    knn = KNeighborsClassifier(n_neighbors = n_neighbors)
    knn.fit(X_train,Y_train.ravel())
    traing_acc.append(knn.score(X_train,Y_train))
    test_acc.append(knn.score(X_test, Y_test))

plt.plot(neighbors_setting, traing_acc, label = "Accuracy of the training set")
plt.plot(neighbors_setting, test_acc, label = "Accuracy of the test set")
plt.ylabel("Accuracuy")
plt.xlabel("neighbors")
plt.grid()
plt.legend()

#.........................................................

#### Improval Model

from sklearn.model_selection import GridSearchCV
parametrs = {"n_neighbors": range(1,20) }
grid_kn = GridSearchCV(estimator = knn, #Model
                       param_grid = parametrs, #Range of K
                       scoring = "accuracy",
                       cv = 5,          # cross validation generator
                       verbose = 1,     #Time of calculate
                       n_jobs = -1)     #help to cpu
                       
            
                     
grid_kn.fit(X_train, Y_train.ravel())

CLF.score(X,Y)

print("Accuracy:" , metrics.accuracy_score(Y_test,Y_pred ))


confusion_matrix(Y,CLF.predict(X))


# Show plot for confusion matrix
cm = confusion_matrix(Y, CLF.predict(X))
fig , ax = plt.subplots(figsize = (8,8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks = (0,1) , ticklabels = ("predicted 0s", "predicted 1s"))
ax.yaxis.set(ticks = (0,1) , ticklabels = ("Actual 0s", "Actual 1s"))
ax.set_ylim(1.5 , -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i , cm[i,j], ha = "center" , va = "center", color = "red")   
plt.show()


# Calculate classification
print(classification_report(Y, CLF.predict(X)))

#CREATING & TRAINING LOGECTIC REGRESSION MODEL

# Sellecting features
X = pd.DataFrame(diabets_df3 , columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"])
Y = diabets_df3.Outcome
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state =0)
Logreg = LogisticRegression(solver = "liblinear")
Logreg.fit(X_train , Y_train)
Y_pred = Logreg.predict(X_test)
Y_pred

#..........................................................

print("Accuracy:" , metrics.accuracy_score(Y_test,Y_pred ))
fpr , tpr,_ = metrics.roc_curve(Y_test,Y_pred)
plt.plot(fpr , tpr, label = "data 1")
plt.legend(loc = 4)
plt.show()

#.........................................................

Y_pred_proba = Logreg.predict_proba(X_test)[::,1]
fpr , tpr,_ = metrics.roc_curve(Y_test,Y_pred_proba)
plt.plot(fpr , tpr, label = "data 1")
plt.legend(loc = 4)
plt.show()

#........................................................

Logreg.classes_
Logreg.intercept_
Logreg.coef_
Logreg.predict_proba(X)
Logreg.score(X,Y)
confusion_matrix(Y, Logreg.predict(X))

#.......................................................

# show plot for confusion matrix
cm = confusion_matrix(Y, Logreg.predict(X))
fig , ax = plt.subplots(figsize = (8,8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks = (0,1) , ticklabels = ("predicted 0s", "predicted 1s"))
ax.yaxis.set(ticks = (0,1) , ticklabels = ("Actual 0s", "Actual 1s"))
ax.set_ylim(1.5 , -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i , cm[i,j], ha = "center" , va = "center", color = "red")   
plt.show()

#......................................................

print(classification_report(Y, Logreg.predict(X)))

#CREATING & TRAINING SVM MODEL

# Sellecting features
X = pd.DataFrame(diabets_df3 , columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"])
Y = diabets_df3.Outcome

# Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25 ,random_state = 1)

#Model
clf_svm = SVC(C=10.0) 
clf_svm = clf_svm.fit(X_train,Y_train )
y_pred = clf_svm.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(Y_test, y_pred))

confusion_matrix(Y, clf_svm.predict(X))

# show plot for confusion matrix
cm = confusion_matrix(Y, clf_svm.predict(X))
fig , ax = plt.subplots(figsize = (8,8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks = (0,1) , ticklabels = ("predicted 0s", "predicted 1s"))
ax.yaxis.set(ticks = (0,1) , ticklabels = ("Actual 0s", "Actual 1s"))
ax.set_ylim(1.5 , -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i , cm[i,j], ha = "center" , va = "center", color = "red")   
plt.show()

#......................................................

#CONCLUSION:Best Algorithm is Logestic Regression

#KNN Accuracy: 0.7402597402597403
#Logestic Regression Accuracy: 0.7708333333333334
#SVM Accuracy: 0.765625





