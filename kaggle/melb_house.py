#https://www.kaggle.com/dansbecker/melbourne-housing-snapshot/downloads/melb_data.csv/5
# Kaggle
import pandas as pd

melbourne_file_path = '../data/melb_data.csv'
df = pd.read_csv(melbourne_file_path) 
print(df.columns)

# dropna drops missing values (think of na as "not available")
# melboune_data is a df
df = df.dropna(axis=0)

# dot Notation to select target variable
# y = melbourne_data.Price
# or using  ['']
y = df['Price']

# choosing features
# right now, we only choose a couple of them in here as examples
mb_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
# by convention, this is called X
X = df[mb_features]
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
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)
# or
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# fit model with train dataset - X_train, y_train
print("split-train\n", mb_model.fit(X_train, y_train))

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

