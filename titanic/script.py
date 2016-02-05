import sys
import numpy as np
#Data manipulation library
import pandas
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation


#Loads the training data
titanic = pandas.read_csv("data/train.csv")

#Loads the testing data
titanic_test = pandas.read_csv("data/test.csv")


#------- Data cleaning ---------
# Training
#--- Age
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

#--- Sex
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
titanic["Sex"] = titanic["Sex"].fillna(0)

#--- Embarked
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2


#Testing 

#Removes the rows with no PassengerId
titanic_test = titanic_test[np.isfinite(titanic_test["PassengerId"])]


#---- PassengerId 
titanic_test["PassengerId"] = titanic_test["PassengerId"].fillna(0)

#---- Pclass
titanic_test["Pclass"] = titanic_test["Pclass"].fillna(titanic["Pclass"].median())

#---- SibSp
titanic_test["SibSp"] = titanic_test["SibSp"].fillna(titanic["SibSp"].median())

#---- Parch
titanic_test["Parch"] = titanic_test["Parch"].fillna(titanic["Parch"].median())

#--- Age
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

#---- Sex 
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
titanic_test["Sex"] = titanic_test["Sex"].fillna(0)


#---- Embarked
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

#---- Fare 
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic["Fare"].median())



# ----- Data enhancement --------

# Generating a familysize column
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]

# The .apply method generates a new series
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))

#print(titanic.describe())
#print(titanic_test.describe())
#sys.exit()


#------ Creates the algorithm
#predictors
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "NameLength"]
#predictors = [ "Sex", "Age"]


# Initialize the algorithm class
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

#Calculates the scores for the training dataframe and pronts it
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
print(scores.mean())

# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set.
predictions = alg.predict(titanic_test[predictors])



# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
    
submission.PassengerId = submission.PassengerId.astype(int)

#Prints the submission    
submission.to_csv("submission.csv", index=False)    



