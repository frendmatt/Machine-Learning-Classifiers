# Support Vector Classification applied on the Wisconsin Breast Cancer Dataset#
# Jeremy Herrmann
# Matthew Jones

# Importing Libraries#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as correlation_plotter
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA


def load_dataset():
    data = pd.read_csv("breast-cancer-wisconsin.csv")
    return data


def remove_invalid_rows(data):
    # The only column where there are invalid cells are in the "Bare Nuclei" column.
    data = data[data['Bare Nuclei'] != "?"]
    return data


def generate_correlation_matrix(data, variables):
    correlation_matrix = data[variables].corr().round(2)
    correlation_plotter.heatmap(data=correlation_matrix, annot=True)
    plt.title('Correlation Matrix', fontsize=36)
    plt.show()


def split_dataset(data, independent_variables, response_variable):
    X = data[independent_variables]
    y = data[response_variable]
    return X, y


def partition_train_and_test(X, y, percent_test):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
    return X_train, X_test, y_train, y_test


def fit_data(X, y):
    svclassifier = SVC(kernel='linear', degree=3)
    # svclassifier = SVC(kernel='poly', gamma = 8)
    # svclassifier = SVC(kernel='rbf', gamma = 1.0)
    svclassifier.fit(X, y)
    return svclassifier


def perform_analysis(Classifier, X, Y):
    y_pred = Classifier.predict(X)

    print(confusion_matrix(Y, y_pred))
    print(classification_report(Y, y_pred))


def drop_for_graph(data):
    data = data.drop('Sample Id', axis=1)
    data = data.drop('Clump Thickness', axis=1)
    data = data.drop('Cell Shape Uniformity', axis=1)
    data = data.drop('Marginal Adhesion', axis=1)
    data = data.drop('Single Epithelial Cell Size', axis=1)
    data = data.drop('Bare Nuclei', axis=1)
    data = data.drop('Bland Chromatin', axis=1)
    data = data.drop('Normal Nucleoli', axis=1)
    return data


if __name__ == "__main__":
    # Loading the dataset
    dataset = load_dataset()
    dataset = remove_invalid_rows(dataset)

    # Now we will find which features are important through a correlation matrix
    generate_correlation_matrix(dataset, dataset.columns)

    # After looking at the correlation matrix, we'll select the highest correlation with the output variable,
    # and remove all variables that have height correlation to the independent variable we just selected. We then
    # select Cell Size Uniformity and Mitoses as our selected independent variables we'll use to train the dataset.
    dataset["Bare Nuclei"] = dataset["Bare Nuclei"].apply(lambda x: int(x))
    Independent_Variables = ["Cell Size Uniformity", "Mitoses", "Clump Thickness", "Cell Shape Uniformity",
                             "Marginal Adhesion", "Single Epithelial Cell Size", "Bland Chromatin", "Normal Nucleoli",
                             "Bare Nuclei"]
    Dependent_Variable = "Class"
    X, y = split_dataset(dataset, Independent_Variables, Dependent_Variable)

    X_Train, X_Test, Y_Train, Y_Test = partition_train_and_test(X, y, 0.3)
    classifier = fit_data(X_Train, Y_Train)

    perform_analysis(classifier, X_Test, Y_Test)
    dataset = drop_for_graph(dataset)
    plt.figure(0)
    correlation_plotter.pairplot(dataset, hue="Class", palette="husl", diag_kind="hist")
    plt.title('MVN Simulated Data', color='#0000ff')
    plt.show()

    # 2D Version
    pca = PCA(n_components=2)
    Xreduced = pca.fit_transform(X)
    X_Train, X_Test, Y_Train, Y_Test = partition_train_and_test(Xreduced, y, 0.3)
    classifier = fit_data(X_Train, Y_Train)

    # Code for showing decision regions
    plot_decision_regions(np.array(X_Train), np.array(Y_Train), clf=classifier, legend=2, )
    plt.figure(1, figsize=(20,20))
    plt.show()

