import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier




X_train=pd.read_csv("/home/shahid/Desktop/titanic_kaggle-master/train.csv")
X_test=pd.read_csv("/home/shahid/Desktop/titanic_kaggle-master/test.csv")
y_train=X_train['Survived'].copy()


z_test=X_test['PassengerId'].copy()

#dropping passengerid
X_train.drop(['PassengerId'], axis=1, inplace=True)
X_test.drop(['PassengerId'], axis=1, inplace=True)


#combining family
pd.crosstab(X_train['Survived'], X_train['Sex'], normalize='index')
X_train['fam']=X_train['SibSp']+X_train['Parch']+1
X_test['fam']=X_test['SibSp']+X_test['Parch']+1


#tabular form study of number of family members and their survival probability
pd.crosstab(X_train['fam'], X_train['Survived'], normalize='index')

#tabular form study of class and survival
pd.crosstab(X_train['Pclass'], X_train['Survived'], normalize='index')



#dropping cabin
X_train.drop(['Cabin'], axis=1, inplace=True)
X_test.drop(['Cabin'], axis=1, inplace=True)

#dropping ticket
X_train.drop(['Ticket'], axis=1, inplace=True)
X_test.drop(['Ticket'], axis=1, inplace=True)


#fillling missing ages
X_train['Age']=X_train['Age'].interpolate()
X_test['Age']=X_test['Age'].interpolate()


#filling missing fare values
X_test.fillna(X_test['Fare'].mean(), axis=1, inplace=True)


#filling missing embarked data
X_train['Embarked'].fillna("S", inplace=True)
X_test['Embarked'].fillna("S", inplace=True)

#after combining family dropping the individual columns
X_train.drop(['Parch'], axis=1, inplace=True)
X_train.drop(['SibSp'], axis=1, inplace=True)

X_test.drop(['Parch'], axis=1, inplace=True)
X_test.drop(['SibSp'], axis=1, inplace=True)

#dropping the name ( although it can be useful )
X_train.drop(['Name'], axis=1, inplace=True)
X_test.drop(['Name'], axis=1, inplace=True)


#changing the non-numeric value to the numeric ones
X_train.loc[X_train['Sex']=='male', 'Sex']=0
X_train.loc[X_train['Sex']=='female', 'Sex']=1

X_train[['Emb_num1','Emb_num2']]=pd.get_dummies(X_train.Embarked).iloc[:,1:]


#same for testing set
X_test.loc[X_test['Sex']=='male', 'Sex']=0
X_test.loc[X_test['Sex']=='female', 'Sex']=1

X_test[['Emb_num1','Emb_num2']]=pd.get_dummies(X_test.Embarked).iloc[:,1:]


#dropping the non-numeric values
X_train.drop(['Embarked'], axis=1, inplace=True)
X_test.drop(['Embarked'], axis=1, inplace=True)


#dropping the target value 
X_train.drop(['Survived'], axis=1, inplace=True)


#models
lr=LogisticRegression()
svm=SVC(kernel='poly', degree=2)
dtc=DecisionTreeClassifier()

#
#rfc=RandomForestClassifier(n_estimators=100, oob_score=True, max_features=5)
#
#rfc.fit(X_train, y_train)
#pred=rfc.predict(X_test)
#print(pred.mean())
#print(rfc.score(X_train, y_train))


#bagging done by voting
#evc=VotingClassifier( estimators = [ ('lr', lr ), ('svm', svm ), ('dtc', dtc) ],voting='hard' )
#evc.fit(X_train, y_train)
#print(evc.score(X_train, y_train))
#pred=evc.predict(X_test)
#

mlp=MLPClassifier(activation="logistic", solver="lbfgs")
mlp.fit(X_train, y_train)
pred=mlp.predict(X_test)
print(mlp.score(X_train, y_train))



#completing the submission file
submissions=pd.DataFrame({
        "PassengerId" : z_test,
        "Survived" : pred
        })

submissions.to_csv("kaggle.csv",sep=',', index=False)


