
# Importing necessary libraries

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 16:46:56 2018
​
@author: Tanay
"""
import os
import time
#import pydot
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.metrics import scorer
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit, train_test_split



# Reading data and observing summary statistics
data = pd.read_csv(r'C:\Users\Tanay\Desktop\Genesis\Boston Housing dataset\BostonHousingDataset.csv')
data.head(5) 
features = data.iloc[:, :-1].values
prices = data.iloc[:, 13].values
data.head()
print('The shape of our features is:', data.shape)
data.describe() # Descriptive statistics for each column


# Statistics of Label (Price)
# Minimum price of the data
minimum_price = np.min(prices)

# Maximum price of the data
maximum_price = np.max(prices)

# Mean price of the data
mean_price = np.mean(prices)

# Median price of the data
median_price = np.median(prices)

# Standard deviation of prices of the data
std_price = np.std(prices)


# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price))
print("Maximum price: ${:,.2f}".format(maximum_price))
print("Mean price: ${:,.2f}".format(mean_price))
print("Median price ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))



# Creating feature list for Decision tree vizualization
col_names = list(data.columns)
feature_list=col_names[0:13]
print(feature_list)



# Training and Testing Sets
# Splitting the data into training and testing sets
train_features, test_features, train_prices, test_prices = train_test_split(features, prices, test_size = 0.25, random_state = 30)
print('Training Features Shape:', train_features.shape)
print('Training Prices Shape:', train_prices.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Prices Shape:', test_prices.shape)


# Establishing a baseline model
# Baseline error: Error generated if we simply predicted the prices for all entries.
baseline_preds = test_features[:, 12]
# Display the average baseline error
baseline_errors = abs(baseline_preds - test_prices)
print('Average baseline error:', round(np.mean(baseline_errors), 2),'dollars.')



#Train the Model
# Instantiating model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 30)
# Train the model on training data
simple_model= rf.fit(train_features, train_prices);


# Making Predictions on the Test Set
# Making predictions on test data
base_predictions = simple_model.predict(test_features)
# Calculating absolute error
base_errors = abs(base_predictions - test_prices)
# Printing mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(base_errors), 2), 'dollars.')



# Determining Performance Metrics

# Calculating mean absolute percentage error (MAPE)
mape = 100 * (base_errors / test_prices)
# Calculating and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 5), '%.')

"""
# Visualizing a Single Decision Tree (Working in Jupyter Notebook)

# Setting path
os.environ["PATH"] += os.pathsep + r'C:\Users\Tanay\Anaconda3\pkgs\graphviz-2.38-hfd603c8_2\Library\bin\graphviz'
os.environ["PATH"] += os.pathsep + r'C:\Users\Tanay\Anaconda3\pkgs\graphviz-2.38.0-4\Library\bin\graphviz'

# Pulling out one tree from the forest
tree = rf.estimators_[5]

# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)

# Using dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')

# Write graph to a png file
graph.write_png('tree.png');

print('The depth of this tree is:', tree.tree_.max_depth)

# Limiting the depth to 2 levels for better viz
rf_small = RandomForestRegressor(n_estimators=1000, max_depth = 3, random_state=30)
rf_small.fit(train_features, train_prices)

# Extracting small tree
tree_small = rf_small.estimators_[5]

# Saving the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
​
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
​
graph.write_png('small_tree.png');
"""

# Doing sample predictions
client_data = [[0.00632, 18, 2.31, 0, 0.538, 6.575, 65.2 ,4.09, 1, 296, 15.3, 396.9, 4.98],
               [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 15, 0, 17],
               [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 22, 0, 32],
               [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 12, 0, 3]]



# Base model predictions
for i, price in enumerate(simple_model.predict(client_data)):
    print ("Predicted selling price for Client {}'s home by using Random forest is: ${:,.2f}".format(i+1, price))


# Model Improvement
# Feature Reduction
# Reducing the number of features to reduce runtime, without significantly reducing performance.
# Feature Importances
# Getting numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sorting feature importances in ascending order (most important)
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];



# Visualize Feature Importances
# Reset style 
# plt.style.use('ggplot')
# plt.style.use('seaborn-bright')
plt.style.use('seaborn-darkgrid')

# list of x locations for plotting
x_values = list(range(len(importances)))

# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical', color = 'g', edgecolor = 'r', linewidth = 1.2)

# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')

# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# List of features sorted from most to least important
sorted_importances = [importance[1] for importance in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]

# Cumulative importances
cumulative_importances = np.cumsum(sorted_importances)

# Make a line graph
plt.plot(x_values, cumulative_importances, 'g-')

# Draw line at 95% of importance retained
plt.hlines(y = 0.95, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')

# Format x ticks and labels
plt.xticks(x_values, sorted_features, rotation = 'vertical')

# Axis labels and title
plt.xlabel('Variable'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importances');

# Limit Number of Features
# Finding number of features for cumulative importance of 95%
# Adding 1 as Python is zero-indexed
print('Number of features for 95% importance:', np.where(cumulative_importances > 0.95)[0][0] + 1)



# Extracting the names of the most important features
important_feature_names = [feature[0] for feature in feature_importances[0:7]]
# Finding columns of the most important features
important_indices = [feature_list.index(feature) for feature in important_feature_names]

# Creating training and testing sets with only important features
important_train_features = train_features[:, important_indices]
important_test_features = test_features[:, important_indices]

# Sanity check on operations
print('Important train features shape:', important_train_features.shape)
print('Important test features shape:', important_test_features.shape)



# Training on Important Features
# Training the model on only the important features
rf.fit(important_train_features, train_prices);

# Evaluate on Important features
# Make predictions on test data
predictions = rf.predict(important_test_features)

# Performance metrics
errors = abs(predictions - test_prices)

print('Average absolute error:', round(np.mean(errors), 4), 'dollars.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_prices)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 5), '%.')


# Comparing Trade-Offs
# Using only the 7 most important features (out of 13) results in increase in accuracy as well as decrease in run time (In my model).
# Use time library for run time evaluation
import time

# All features training and testing time
all_features_time = []

# Do 10 iterations and take average for all features
for _ in range(10):
    start_time = time.time()
    rf.fit(train_features, train_prices)
    all_features_predictions = rf.predict(test_features)
    end_time = time.time()
    all_features_time.append(end_time - start_time)
all_features_time = np.mean(all_features_time)
print('All features total training and testing time:', round(all_features_time, 2), 'seconds.')


# Total training and testing time for reduced feature set
reduced_features_time = []

# Doing 10 iterations and taking average
for _ in range(10):
    start_time = time.time()
    rf.fit(important_train_features, train_prices)
    reduced_features_predictions = rf.predict(important_test_features)
    end_time = time.time()
    reduced_features_time.append(end_time - start_time)
    
    reduced_features_time = np.mean(reduced_features_time)
print('Reduced features total training and testing time:', round(reduced_features_time, 2), 'seconds.')


#  Accuracy vs Run-Time
all_accuracy =  100 * (1- np.mean(abs(all_features_predictions - test_prices) / test_prices))
reduced_accuracy = 100 * (1- np.mean(abs(reduced_features_predictions - test_prices) / test_prices))

comparison = pd.DataFrame({'features': ['all (13)', 'reduced (7)'], 
                           'run_time': [round(all_features_time, 2), round(reduced_features_time, 5)],
                           'accuracy': [round(all_accuracy, 2), round(reduced_accuracy, 6)]})
comparison[['features', 'accuracy', 'run_time']]


relative_accuracy_increase = 100 * (all_accuracy - reduced_accuracy) / all_accuracy
print('Relative increase in accuracy:', round(relative_accuracy_increase, 10), '%.')

relative_runtime_decrease = 100 * (all_features_time - reduced_features_time) / all_features_time
print('Relative decrease in run time:', round(relative_runtime_decrease, 5), '%.')



# Hyper parameter Tuning
# Examining the default Random Forest to determine parameters
rf = RandomForestRegressor(random_state = 30)
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())



# Random Search with Cross Validation

# Random Hyperparameter Grid
# Creating a parameter grid to sample from during fitting
    
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)  



# Random Search Training
# Instantiating the random search and fitting it.
# Using the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=30, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_features, train_prices)    
rf_random.best_params_



# Evaluating Random Search
# If random search yielded a better model, we compare the base model with the best random search model
# Evaluation Function
def evaluate(model, test_features, test_prices):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_prices)
    mape = 100 * np.mean(errors / test_prices)
    accuracy = 100 - mape
    print('Model Performance:')
    print('Average Error: {:0.3f} dollars.'.format(np.mean(errors)))
    print('Accuracy = {:0.10f}%.'.format(accuracy))    
    return accuracy

# Evaluating the Default (Base) Model
base_model = RandomForestRegressor(n_estimators = 1000, random_state = 30)
base_model.fit(train_features, train_prices)
base_accuracy = evaluate(base_model, test_features, test_prices)


# Evaluating the Best Random Search Model
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_prices)
print('Improvement of {:0.6f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))


# Grid Search with Cross Validation
#Performing grid search building on the result from the random search. 
# And testing a range of hyperparameters around the best values returned by random search.
# Making another grid based on the best values provided by random search:

param_grid = {
    'bootstrap': [False],
    'max_depth': [110,None],
    'max_features': [2, 3],
    'min_samples_leaf': [1],
    'min_samples_split': [2],
    'n_estimators': [400]
}
# Creating a based model
rf = RandomForestRegressor()
# Instantiating the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fitting the grid search to the data
grid_search.fit(train_features, train_prices)

# Fitting the model, display the best hyperparameters, and evaluate performance:
grid_search.best_params_

# Evaluating the Best Model from Grid Search
best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, test_features, test_prices)    
print('Improvement of {:0.5f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))


# Another Round of Grid Search
# Create the parameter grid based on the results of previous Grid search

param_grid = {
    'bootstrap': [False],
    'max_depth': [None],
    'max_features': [3],
    'min_samples_leaf': [1],
    'min_samples_split': [2],
    'n_estimators': [400]
}
# Creating a based model
rf = RandomForestRegressor()
# Instantiating the grid search model
grid_search_ad = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search_ad.fit(train_features, train_prices)
grid_search_ad.best_params_
best_grid_ad= grid_search_ad.best_estimator_
grid_ad_accuracy=evaluate(best_grid_ad, test_features, test_prices)
print('Improvement of {:0.5f}%.'.format( 100 * (grid_ad_accuracy - grid_accuracy) / base_accuracy))

# This time our performance slightly increased. Therefore, finalizing this model
# Final Model
print('Model Parameters:\n')
pprint(grid_search.get_params())
print('\n')
evaluate(grid_search_ad, test_features, test_prices)


# Training Visualizations

# Training Curves
plt.style.use('seaborn-darkgrid')

# Grid with only the number of trees changed
tree_grid = {'n_estimators': [int(x) for x in np.linspace(1, 301, 30)]}

# Create the grid search model and fit to the training data
tree_grid_search = GridSearchCV(best_grid_ad, param_grid=tree_grid, verbose = 2, n_jobs=-1, cv = 3,
                                scoring = 'neg_mean_absolute_error')
tree_grid_search.fit(train_features, train_prices);
tree_grid_search.cv_results_


def plot_results(model, param = 'n_estimators', name = 'Num Trees'):
    param_name = 'param_%s' % param
    # Extract information from the cross validation model
    train_scores = model.cv_results_['mean_train_score']
    test_scores = model.cv_results_['mean_test_score']
    train_time = model.cv_results_['mean_fit_time']
    param_values = list(model.cv_results_[param_name])
    
    # Plot the scores over the parameter
    plt.subplots(1, 2, figsize=(10, 6))
    plt.subplot(121)
    plt.plot(param_values, train_scores, 'bo-', label = 'train')
    plt.plot(param_values, test_scores, 'go-', label = 'test')
    plt.ylim(ymin = -10, ymax = 10)
    plt.legend()
    plt.xlabel(name)
    plt.ylabel('Neg Mean Absolute Error')
    plt.title('Score vs %s' % name)    
    plt.subplot(122)
    plt.plot(param_values, train_time, 'ro-')
    plt.ylim(ymin = -10.0, ymax = 10.0)
    plt.xlabel(name)
    plt.ylabel('Train Time (sec)')
    plt.title('Training Time vs %s' % name)        
    plt.tight_layout(pad = 4)
        
plot_results(tree_grid_search)

# Number of Features at Each Split
# Define a grid over only the maximum number of features
feature_grid = {'max_features': list(range(1, train_features.shape[1] + 1))}

# Creating the grid search and fitting on the training data
feature_grid_search = GridSearchCV(best_grid, param_grid=feature_grid, cv = 3, n_jobs=-1, verbose= 2,
                                  scoring = 'neg_mean_absolute_error')
feature_grid_search.fit(train_features, train_prices);

plot_results(feature_grid_search, param='max_features', name = 'Max Features')


# Tuned model predictions
for i, price in enumerate(grid_search_ad.predict(client_data)):
    print ("Predicted selling price for Client {}'s home by using Random forest is: ${:,.2f}".format(i+1, price))


# Save the model to disk
filename = r'C:\Users\Tanay\Desktop\Genesis\Final project\Jupyter notebook\RandomForest.sav'
pickle.dump(grid_search_ad, open(filename, 'wb'))
 
# loading the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(test_features, test_prices)
#print(result)​