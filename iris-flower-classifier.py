import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
# Feature columns
#   1 = sepal length
#   2 = sepal width
#   3 = petal length
#   4 = petal width
# Target columns
#   1 = setosa (starts at index 0)
#   2 = versicolor (starts at index 50)
#   3 = virginica (starts at index 100)
iris = load_iris()
test_index = [0, 50, 100]

# Training data
train_data = np.delete(iris.data, test_index, axis=0)
train_target = np.delete(iris.target, test_index)


# Testing data
test_data = iris.data[test_index]
test_target = iris.target[test_index]

clsfr = tree.DecisionTreeClassifier()
clsfr = clsfr.fit(train_data, train_target)

print(test_index)
print(test_target)
print(clsfr.predict(test_data))
