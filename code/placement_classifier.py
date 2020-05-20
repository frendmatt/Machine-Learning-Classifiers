### CSE 353 Final Project ###
###    Jeremy Herrmann    ###
###     Matthew Jones     ###


'''ML project attempting to classify whether or not a college student
   will be placed with a job, based on the college recruitment data set'''

###  Imports ###
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA


import matplotlib.pyplot as plt 
import matplotlib


### /Imports ###


def process_dataset(dataset):
    dataset['gender'] = dataset['gender'].replace("M", 0)
    dataset['gender'] = dataset['gender'].replace("F", 1)
    dataset['ssc_p'] = dataset['ssc_p'].apply(lambda x: (x / 100))
    dataset['hsc_p'] = dataset['hsc_p'].apply(lambda x: (x / 100))
    dataset['ssc_b'] = dataset['ssc_b'].replace("Central", 0)
    dataset['ssc_b'] = dataset['ssc_b'].replace("Others", 1)
    dataset['hsc_b'] = dataset['hsc_b'].replace("Central", 0)
    dataset['hsc_b'] = dataset['hsc_b'].replace("Others", 1)
    dataset['Arts'] = dataset['hsc_s'].replace("Arts", 1)
    dataset['Arts'] = dataset['Arts'].replace("Commerce", 0)
    dataset['Arts'] = dataset['Arts'].replace("Science", 0)
    dataset['Science'] = dataset['hsc_s'].replace("Science", 1)
    dataset['Science'] = dataset['Science'].replace("Arts", 0)
    dataset['Science'] = dataset['Science'].replace("Commerce", 0)
    dataset['Commerce'] = dataset['hsc_s'].replace("Commerce", 1)
    dataset['Commerce'] = dataset['Commerce'].replace("Science", 0)
    dataset['Commerce'] = dataset['Commerce'].replace("Arts", 0)
    dataset = dataset.drop(["hsc_s"], axis=1)
    dataset['degree_p'] = dataset['degree_p'].apply(lambda x: (x / 100))
    dataset = dataset.drop(columns=['degree_t'])
    dataset['workex'] = dataset['workex'].replace("No", 0)
    dataset['workex'] = dataset['workex'].replace("Yes", 1)
    dataset['etest_p'] = dataset['etest_p'].apply(lambda x: (x / 100))
    dataset['specialisation'] = dataset['specialisation'].replace("Mkt&Fin", 0)
    dataset['specialisation'] = dataset['specialisation'].replace("Mkt&HR", 1)
    dataset['mba_p'] = dataset['mba_p'].apply(lambda x: (x / 100))
    dataset['status'] = dataset['status'].replace("Placed", 0)
    dataset['status'] = dataset['status'].replace("Not Placed", 1)

    return dataset


def load_dataset():
    dataset = pd.read_csv("./Placement_Data_Full_Class.csv")
    dataset = process_dataset(dataset)
    return dataset


def split_dataset(dataset, independent_variables, response_variable):
    X = dataset[independent_variables]
    y = dataset[response_variable]
    return X, y


def partition_train_and_test(X, y, testing_percentage):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing_percentage, shuffle=True)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Getting the dataset + fixing it up
    dataset = load_dataset()
    
    # Setting up our features + class
    independent_variables = ['ssc_p', 'hsc_p', 'degree_p', 'workex', 'etest_p', 'mba_p']
    response_variable = 'status'

    # Splitting the data and making a test/train set
    X, Y = split_dataset(dataset, independent_variables, response_variable)
    X_train, X_test, Y_train, Y_test = partition_train_and_test(X, Y, 0.3)
    
    # Checking which value of K work the best
    Accuracies = {}
    for k in range(1, 41):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, Y_train)

        predicted_y = model.predict(X_test)

        scores = cross_val_score(model, X, Y, cv = 5)
        average = sum(scores) / len(scores) * 100
        Accuracies[k] = average
        print("{}-Neighbor 5-Fold CV gives us an average accuracy of {}%".format(k, average))

    # Plotting this accuracy
    x = Accuracies.keys()
    y = Accuracies.values()
    plt.scatter(x, y, color="green")
    plt.title('Accuracy Percentage versus K')
    plt.show()
    
    # Plot of the features with the classified points
    for feature in X:
        x_placed = []
        y_placed = []
        
        x_not_placed = []
        y_not_placed = []

        for index in range(0, len(Y)):
            if Y[index] == 0:
                x_placed.append(X[feature][index])
                y_placed.append(Y[index])
            else:
                x_not_placed.append(X[feature][index])
                y_not_placed.append(Y[index])
                
        plt.scatter(x_placed, y_placed, color="green")
        plt.scatter(x_not_placed, y_not_placed, color="red")
        plt.title('Plot of {} with classified points'.format(feature))
        plt.show()
    
    # Plotting the features against each other
    features = X.columns
    for i in range(0, len(features)):
        for j in range(i, len(features)):
            plt.scatter(X[features[i]], X[features[j]], alpha=0.3,
                s=100, c=Y, cmap='viridis')
            plt.xlabel(features[i])
            plt.ylabel(features[j])
            plt.title('{} versus {}'.format(features[j], features[i]))
            plt.show()
    
    
    # PCA ANALYSIS & VISUALIZATION###

    # Converting the data to two-dimensional using PCA
    pca_model = PCA(n_components=2)
    pca_model.fit(X_train)
    X_train = pca_model.transform(X_train)
    X_test = pca_model.transform(X_test)
    
    # Checking which value of K works the best
    Accuracies = {}
    for k in range(1, 41):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, Y_train) 

        predicted_y = model.predict(X_test)
        
        scores = cross_val_score(model, X, Y, cv = 5)
        average = sum(scores) / len(scores) * 100
        Accuracies[k] = average
        print("{}-Neighbor 5-Fold CV gives us an average accuracy of {}%".format(k, average))

    # Plotting this accuracy
    x = Accuracies.keys()
    y = Accuracies.values()
    plt.scatter(x, y, color="green")
    plt.legend(Accuracies.keys())
    plt.show()

    # Training the model with the ideal value of K = 7
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(X_train, Y_train) 

    # Plotting the decision boundaries from the model
    # Setting up the plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    padding = 0.1
    resolution = 0.1
    # Finding the max-min values to plot
    colors = {0: 'green', 1: 'red'}
    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
    y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding
    y_min -= y_range * padding
    x_max += x_range * padding
    y_max += y_range * padding
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.BrBG)
    plt.axis('tight')

    # Plot your testing points as well
    for label in np.unique(Y_test):
        indices = np.where(Y_test == label)
        plt.scatter(X_test[indices, 0], X_test[indices, 1], c=colors[label], alpha=0.8, label=("Placed" if label == 0 else "Not - Placed"))

    plt.legend(loc='lower right')
    plt.title('PCA Decision Boundaries using 7-NN')
    plt.show()

    



