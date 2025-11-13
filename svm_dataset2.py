import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def calc_lower_bound(mean, std):
    return mean - ((1.96 * std) / np.sqrt(10))


def calc_upper_bound(mean, std):
    return mean + ((1.96 * std) / np.sqrt(10))


dataset, labels = pickle.load(open("../data/part2_dataset2.data", "rb"))

pipeline = Pipeline([('scaler', StandardScaler()),
                     ('svm', SVC())])

parameter_grid = {"svm__kernel": ['linear', 'poly', 'rbf'],
                  "svm__C": [0.01, 100]}

cross_validation = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)
# define the grid search we will be using and setting the scoring to the accuracy
grid_search = GridSearchCV(pipeline, parameter_grid, scoring='accuracy', cv=cross_validation)

grid_search.fit(dataset, labels)

parameters = grid_search.cv_results_["params"]
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']

for i in range(len(parameters)):
    print("Statistics for Kernel: %s - C: %f" % (parameters[i]["svm__kernel"], parameters[i]["svm__C"]))
    print("Mean: %.2f" % means[i])
    print("Std: %.2f" % stds[i])
    print("Lower Bound: %.2f" % calc_lower_bound(means[i], stds[i]))
    print("Upper Bound: %.2f" % calc_upper_bound(means[i], stds[i]))
    print("------------------------------------------------------------")


i = grid_search.best_index_

print("Best Hyperparameter Kernel: %s - C: %f" % (parameters[i]["svm__kernel"], parameters[i]["svm__C"]))
print("Mean: %.2f" % means[i])
print("Std: %.2f" % stds[i])
print("Lower Bound: %.2f" % calc_lower_bound(means[i], stds[i]))
print("Upper Bound: %.2f" % calc_upper_bound(means[i], stds[i]))
print("------------------------------------------------------------")


