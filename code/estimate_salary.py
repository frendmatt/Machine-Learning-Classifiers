### CSE 353 Final Project ###
###    Jeremy Herrmann    ###
###     Matthew Jones     ###


'''ML project attempting to classify whether or not a college student
   will be placed with a job, based on the college recruitment data set'''

# API for arrays
import numpy as np

# API for plotting
import matplotlib.pyplot as plt

# API for excel data
import pandas as pd

# API for statistics
import statsmodels.api as sm
from statsmodels.formula.api import ols

# API for plotting correlation matrix
import seaborn as correlation_plotter


def process_dataset(dataset):
    dataset['gender'] = dataset['gender'].replace("M", 0)
    dataset['gender'] = dataset['gender'].replace("F", 1)

    dataset['ssc_p'] = dataset['ssc_p'].apply(lambda x: (x / 100))
    dataset['hsc_p'] = dataset['hsc_p'].apply(lambda x: (x / 100))  # Changed hsc_p from ssc_p

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

    dataset['degree_p'] = dataset['degree_p'].apply(lambda x: (x / 100))  # Changed degree_p from ssc_p

    dataset = dataset.drop(columns=['degree_t'])

    dataset['workex'] = dataset['workex'].replace("No", 0)
    dataset['workex'] = dataset['workex'].replace("Yes", 1)

    dataset['etest_p'] = dataset['etest_p'].apply(lambda x: (x / 100))

    dataset['specialisation'] = dataset['specialisation'].replace("Mkt&Fin", 0)
    dataset['specialisation'] = dataset['specialisation'].replace("Mkt&HR", 1)

    dataset['mba_p'] = dataset['mba_p'].apply(lambda x: (x / 100))

    dataset['status'] = dataset['status'].replace("Placed", 0)
    dataset['status'] = dataset['status'].replace("Not Placed", 1)
    dataset['salary'] = dataset['salary'].apply(lambda x: (x / 100000 ))

    return dataset


def get_data():
    dataset = pd.read_csv("./Placement_Data_Full_Class.csv")
    dataset = process_dataset(dataset)
    return dataset


def generate_regression_statistics(x, y):
    results = sm.OLS(y, x).fit()
    print(results.summary())


def generate_regression_residuals(data, x_title, y_title):
    model = ols(y_title + " ~ " + x_title, data).fit()
    fig = plt.figure(figsize=(12, 8))
    sm.graphics.plot_regress_exog(model, x_title, fig=fig)
    plt.show()


def generate_correlation_matrix(data, variables):
    correlation_matrix = data[variables].corr().round(2)
    correlation_plotter.heatmap(data = correlation_matrix, annot=True)
    plt.title('Correlation Matrix', fontsize=36)
    plt.show()


def create_variable_coefficients(x, y):
    # Converting to np array
    x = np.array(x)
    y = np.array(y)
    
    # Getting the coefficients of the matrix
    # X Transpose
    x_t = x.transpose()
    # X Transpose * X
    x_t_x = np.matmul(x_t, x)
    # (X Transpose * X)^-1
    x_t_x_inv = np.linalg.inv(x_t_x)
    # (X Transpose * X)^-1 * X Transpose
    x_t_x_inv_x_t = np.matmul(x_t_x_inv, x_t)
    # (X Transpose * X)^-1 * X Transpose * Y
    B = x_t_x_inv_x_t_y = np.matmul(x_t_x_inv_x_t, y)

    # Returning the new x, y, and coefficient matrices / vectors
    return x, y, B


def multiple_linear_regression_main():
    data = get_data()
    data = data.dropna()
    data = data.reset_index(drop=True)

    # Setting up the initial independent variable list and performing MLR, printing our the results
    Independent_Variable_List = ['gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'Arts', 'Science', 'Commerce', 'degree_p', 'workex', 'etest_p', 'specialisation', 'mba_p']
    x = data[Independent_Variable_List]
    y = data['salary']
    X, Y, coefficients = create_variable_coefficients(x.copy(), y.copy())
    # Printing out the stats for this model
    X, Y, coefficients = create_variable_coefficients(x.copy(), y.copy())
    Independent_Variable_List = ['gender', 'hsc_p', 'Arts',
                                  'workex', 'etest_p', 'specialisation', 'mba_p']
    x = data[Independent_Variable_List]
    y = data['salary']
    generate_regression_statistics(x, y)
    
    # Generating the correlation matrix -- This will show us which variables have high correlation
    # We will then remove variable with high correlation to eachother, keeping the variable
    # with the highest correlation to salary
    # This is called feature reduction
    generate_correlation_matrix(data, Independent_Variable_List + ["salary"])

    # Now we will perform our analysis on the model
    # 1. The coefficient of determination is given from the generate_regression_statistics method.
    # 2. Similarly, the generate_regression_statistics method performs an F-test on the model and
    # t-tests on each of the variables.
    # 3. Then we will go ahead and calculate the residuals for this model, along with the mean, and
    # displaying them to show they follow a normal distribution with mean = 0. We will also calculate the noise variance
    estimates = [0 for i in range(0, len(x))]
    for i in range(0, len(estimates)):
        for j in range(0, len(Independent_Variable_List)):
            estimates[i] += x[Independent_Variable_List[j]][i] * coefficients[j]
    residuals = [(estimates[i] - data['salary'][i]) for i in range(0, len(x))]
    data['ESTIMATE'] = estimates
    data['RESIDUALS'] = residuals

    SUM = data['RESIDUALS'].sum()
    AVERAGE = SUM / data['RESIDUALS'].count()
    print("Residual Mean: " + str(AVERAGE))

    SSE = 0
    for residual in data['RESIDUALS']:
        SSE += residual ** 2
    NOISE_VARIANCE = SSE / (data['RESIDUALS'].count() - len(Independent_Variable_List))
    Y_BAR = data['salary'].sum()
    COEFFICIENT_OF_VARIANCE = NOISE_VARIANCE / Y_BAR * 100
    print("Coefficient Of Variance " + str(COEFFICIENT_OF_VARIANCE))

    _ = plt.hist(data['RESIDUALS'], bins=35)  # arguments are passed to np.histogram
    plt.title('Residual Distribution', fontsize=36)
    plt.show()
    
    # 4. Finally, we will plot the residuals against each of the
    # independent variables and the dependent variable itself.
    for var in (Independent_Variable_List + ['salary']):
        generate_regression_residuals(data, var, 'RESIDUALS')


if __name__ == "__main__":
    multiple_linear_regression_main()








    





    
