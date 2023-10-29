import sys
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics # To check the accuracy of the model
import matplotlib.pyplot as plt 

# Decision Tree to predict a drug for a new pacient
# Load the data 
my_data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv', delimiter=",")
my_data.head()

# Removed the target name
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]


# Convert these features to numerical values using pandas.get_dummies()
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]

# Fill the target variable
y = my_data["Drug"]
y[0:5]



# The X and y are the arrays required before the split, 
# The test_size represents the ratio of the testing dataset, 
# The random_state ensures that we obtain the same splits.
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
  

# Inside of the classifier, specified "entropy" so we can see the information gain of each node.
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters

# Fit the data
drugTree.fit(X_trainset,y_trainset)

# Make the predictions
predTree = drugTree.predict(X_testset)

# Visualize and predic
print (predTree [0:5])
print (y_testset [0:5])



# To check the accuracy of the model
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree)) 


tree.plot_tree(drugTree)
plt.show()

#OUTPUT
#40     drugY
#51     drugX
#139    drugX
#197    drugX
#170    drugX
#Name: Drug, dtype: object
#DecisionTrees's Accuracy:  0.9833333333333333 



















