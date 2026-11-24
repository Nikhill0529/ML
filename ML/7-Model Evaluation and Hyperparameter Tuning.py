# This code may not be run on GFG IDE 
# as required packages are not found.
# STRATIFIES K-FOLD CROSS VALIDATION { 10-fold }
# Import Required Modules.
from statistics import mean, stdev 
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold 
from sklearn import linear_model
from sklearn import datasets
# FEATCHING FEATURES AND TARGET VARIABLES IN ARRAY FORMAT.
cancer = datasets.load_breast_cancer() 
# Input_x_Features.
x = cancer.data
# Input_ y_Target_Variable. 
y = cancer.target
# Feature Scaling for input features. 
scaler = preprocessing.MinMaxScaler() 
x_scaled = scaler.fit_transform(x)
# Create classifier object.
lr = linear_model.LogisticRegression() 
# Create StratifiedKFold object.
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) 
lst_accu_stratified = []
for train_index, test_index in skf.split(x, y):
    x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    lr.fit(x_train_fold, y_train_fold)
    lst_accu_stratified.append(lr.score(x_test_fold, y_test_fold))
# Print the output.
print('List of possible accuracy:', lst_accu_stratified)
print('\nMaximum Accuracy That can be obtained from this model is:', 
max(lst_accu_stratified)*100, '%')
print('\nMinimum Accuracy:', 
min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
mean(lst_accu_stratified)*100, '%')
print('\nStandard Deviation is:', stdev(lst_accu_stratified))











from sklearn.model_selection import GridSearchCV, train_test_split
# import train_test_split
from sklearn.tree import DecisionTreeClassifier
# Assuming 'X' and 'y' are your features and target variable respectively 
# from previous cells or loaded from a dataset
X = iris.data # Replace with your actual features
y = iris.target # Replace with your actual target variable 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# split data
param_grid = { 
'max_depth': [3, 5, 7],
'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5) 
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_ 
print(best_params)
