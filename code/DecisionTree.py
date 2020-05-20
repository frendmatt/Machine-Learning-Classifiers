# Support Vector Classification applied on the Wisconsin Breast Cancer Dataset#
# Jeremy Herrmann
# Matthew Jones

# Importing Libraries#
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as correlation_plotter
from sklearn import tree
from sklearn.model_selection import cross_val_score

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus


def load_dataset():
    data = pd.read_csv("TitanicData.csv")
    return data


def clean_dataset(data):
    data = data.drop('PassengerId', axis=1)
    data = data.drop('Name', axis=1)
    data = data.drop('Ticket', axis=1)
    number_rows = data.shape[0]
    # Counting the number of elements missing from each column
    columns_to_remove = []
    for column in data:
        if data[column].isna().sum() / number_rows >= 0.7:
            columns_to_remove.append(column)
    data = data.drop(columns_to_remove, axis=1)
    # The average age is about 29.7 years, and the peak is between 24 - 30 years, so we will replace all the missing
    # values with the mean.
    age_list = [age for age in data['Age'] if str(age) != "nan"]
    average_age = sum(age_list) / len(age_list)
    # Filling in the missing Age values using the Mean Value
    data['Age'] = data['Age'].fillna(average_age)
    # Grouping the Age into equal groups of values
    ages = [age for age in data['Age']]
    ages.sort()
    number_groups = 4
    age_groups = []
    for i in range(number_groups):
        amount_per_group = len(ages) / number_groups
        start_age = ages[int(i * amount_per_group)]
        end_age = ages[int((i + 1) * amount_per_group) - 1]
        age_groups.append((start_age, end_age))
    for i in data['Age'].index:
        for group in range(len(age_groups)):
            if age_groups[group][0] <= data['Age'][i] <= age_groups[group][1]:
                data['Age'].at[i] = group
    # Grouping the Fares into equal groups of values
    fares = [fare for fare in data['Fare']]
    fares.sort()
    number_groups = 4
    fare_groups = []
    for i in range(number_groups):
        amount_per_group = len(fares) / number_groups
        start_fare = fares[int(i * amount_per_group)]
        end_fare = fares[int((i + 1) * amount_per_group) - 1]
        fare_groups.append((start_fare, end_fare))
    for i in data['Fare'].index:
        for group in range(len(fare_groups)):
            if fare_groups[group][0] <= data['Fare'][i] <= fare_groups[group][1]:
                data['Fare'].at[i] = group
    # Dropping the missing rows of Embarked, since we only have two missing values
    data = data[data['Embarked'].notna()]
    # Replacing all male's with 0 and females with 1
    data['Sex'] = data['Sex'].replace('male', 0)
    data['Sex'] = data['Sex'].replace('female', 1)
    # Replacing all the Embarked categorical values with numerical values
    data['Embarked'] = data['Embarked'].replace('S', 0)
    data['Embarked'] = data['Embarked'].replace('Q', 1)
    data['Embarked'] = data['Embarked'].replace('C', 2)
    return data


def generate_correlation_matrix(data, variables):
    correlation_matrix = data[variables].corr().round(2)
    correlation_plotter.heatmap(data=correlation_matrix, annot=True)
    plt.title('Correlation Matrix', fontsize=36)
    plt.show()


def split_dataset(dataset, independent_variables, response_variable):
    X = dataset[independent_variables]
    y = dataset[response_variable]
    return X, y


def partition_train_and_test(X, y, percent_test):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
    return X_train, X_test, y_train, y_test


def fit_data(X, y, depth):
    clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=depth, random_state=400)
    clf = clf.fit(X, y)
    return clf


def perform_analysis(Classifier, X, Y):
    y_pred = Classifier.predict(X)
    accuracy = accuracy_score(Y, y_pred)
    return (accuracy_score(Y, y_pred))


if __name__ == "__main__":
    # Loading the dataset
    dataset = load_dataset()
    dataset = clean_dataset(dataset)

    # Setting up the initial independent variable list and performing MLR, printing our the results
    generate_correlation_matrix(dataset, dataset.columns)
    print(dataset.columns)

    # Now we will find which features are important through a correlation matrix generate_correlation_matrix(dataset,
    # dataset.columns) After looking at the correlation matrix, we'll select the highest correlation with the output
    # variable, and remove all variables that have height correlation to the independent variable we just selected. We
    # then select Cell Size Uniformity and Mitoses as our selected independent variables we'll use to train the
    # dataset.
    Independent_Variables = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    Dependent_Variable = "Survived"
    X, y = split_dataset(dataset, Independent_Variables, Dependent_Variable)

    X_Train, X_Test, Y_Train, Y_Test = partition_train_and_test(X, y, 0.4)

    classifiers = []
    results = []
    for depth in range(1, 10):
        classifier = fit_data(X_Train, Y_Train, depth)
        classifiers.append(classifier)
        accuracy = perform_analysis(classifier, X_Test, Y_Test)
        results.append((depth, accuracy))

        fn = Independent_Variables
        cn = [Dependent_Variable]

        """fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=600)
        tree.plot_tree(classifiers[depth-1],
                       feature_names = fn, 
                       class_names="Survived",
                       filled = True);
        fig.savefig('Depth' + str(depth) + '.png')"""

    depth = [i[0] for i in results]
    accuracy = [i[1] for i in results]
    plt.scatter(depth, accuracy)
    plt.show()

    scores = cross_val_score(classifier, X, y, cv=5)
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())
