# https://www.kaggle.com/alexisbcook/missing-values
# Kaggle
# missing values/Imputer
import pandas as pd


melbourne_file_path = '../data/melb_data.csv'
df = pd.read_csv(melbourne_file_path) 
print(df.columns)

# dropna drops missing values (think of na as "not available")
# melboune_data is a df
# but this will lead to not acurate prediction - see imputation model

# dot Notation to select target variable or predictor
# y = melbourne_data.Price
# or using  ['']
y = df['Price']

# choosing features or independend vaiables
# by convention, this is called X
# To keep things simple, we'll use only numerical predictors
melb_predictors = df.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])


#X = df[['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']]

# Lets take a quick look of X dataset
print(X.describe())
print(X.head())

# building model
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define model. Specify a number for random_state to ensure same results each run
mb_model = DecisionTreeRegressor(random_state=1)

# data are loaded in X, y in DataFRames
# split data into train and test or validate
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
# or
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define Function to Measure Quality of Each Approach
# We define a function score_dataset() to compare different approaches to 
# dealing with missing values. This function reports the mean absolute error (MAE) 
# from a random forest model.
from sklearn.ensemble import RandomForestRegressor 

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

###
### missing values with SimpleImputer
## 
from sklearn.impute import SimpleImputer

# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# option 1 - Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

# option 2 - using SimpleImputer
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

# We see that Approach 2 has lower MAE than Approach 1, 
# so Approach 2 performed better on this dataset.

# option 3 - Score from Approach 3 (An Extension to Imputation)
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))

print(X_train.shape)
# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

print(X_train_plus.head())

# As is common, imputing missing values (in Approach 2 and Approach 3) yielded better results, 
# relative to when we simply dropped columns with missing values (in Approach 1).

######## original melboune house
########

# dropna drops missing values (think of na as "not available")
# melboune_data is a df
df = df.dropna(axis=0)
# choosing features
# right now, we only choose a couple of them in here as examples
mb_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
y = df['Price']
X = df[mb_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)

# fit model with train dataset - X_train, y_train
print("\nSplit-train\n", mb_model.fit(X_train, y_train))

#print("Making predictions for the following 10 houses:")
#print(X_train.head(10))
#The predictions with X_train
y_pred = mb_model.predict(X_train)
mae = mean_absolute_error(y_train, y_pred)
print("X_train MAE:", mae)
# "X_test-the Predictions are 
y_pred = mb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Y_test MAE:", mae)

# are that MAE good?  There isn't a general rule for what values are good 
# that applies across applications. But you'll see how to use (and improve) 
# this number in the next step.


# Fit model- see the output
#print("whole dataset\n", mb_model.fit(X, y))
"""
Many machine learning models allow some randomness in model training. 
Specifying a number for random_state ensures you get the same results in each run.
This is considered a good practice. You use any number, and model quality won't depend 
meaningfully on exactly what value you choose.
We now have a fitted model that we can use to make predictions.
"""
print("Making predictions for the following 10 houses:")
print(X.head(10))
print("The predictions are")
print(mb_model.predict(X.head(10)))

