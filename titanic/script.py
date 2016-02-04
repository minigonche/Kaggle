#Data manipulation library
import pandas
# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold

#Loads teh data
titanic = pandas.read_csv("data/train.csv")

print(titanic.describe())

#------- Data cleaning ---------
#--- Age
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

#--- Sex
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

#--- Embarked
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2