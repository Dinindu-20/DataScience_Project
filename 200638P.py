
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from datetime import datetime 
import numpy as np

# Load the CSV file into a Data Frame
chatterbox = pd.read_csv('employees.csv')
# Identify the data type of each variable in the Data Frame
data_types = chatterbox.dtypes

# Display the data types
print(data_types)
chatterbox.isna().sum()

print('Marital_Status : ', chatterbox['Marital_Status'].unique())
print('Title : ', chatterbox['Title'].unique())
print('Gender : ', chatterbox['Gender'].unique())
print('Religion_ID : ', chatterbox['Religion_ID'].unique())


for col in chatterbox.columns:
    print(chatterbox[str(col)].unique())

#Considering that Gender is marked correctly. title will be corrected according to Gender and marital status
for index,row in chatterbox.iterrows():
  if row['Marital_Status'] == 'Married' and row['Gender'] == 'Female':
    row['Title'] = 'MS'
  elif row['Marital_Status'] == 'Single' and row['Gender'] == 'Female':
    row['Title'] = 'Miss'
  elif row['Gender'] == 'Male':
    row['Title'] = 'Mr'

chatterbox_temp = chatterbox.drop(['Employee_Code', 'Name', 'Religion_ID', 'Designation_ID', 'Date_Resigned', 'Inactive_Date', 'Reporting_emp_1', 'Reporting_emp_2','Year_of_Birth'], axis=1)

for index, row in chatterbox_temp.iterrows():
    date = row['Date_Joined'].split('/')
    date = datetime(int(date[2]), int(date[0]), int(date[1]))
    chatterbox_temp.at[index, 'Date_Joined_Days'] = int(date.timestamp() / 86400)

label_encoder = preprocessing.LabelEncoder()

chatterbox_temp['Employment_Category'] = label_encoder.fit_transform(chatterbox_temp['Employment_Category'])
chatterbox_temp['Employment_Type'] = label_encoder.fit_transform(chatterbox_temp['Employment_Type'])
chatterbox_temp['Gender'] = label_encoder.fit_transform(chatterbox_temp['Gender'])
chatterbox_temp['Title'] = label_encoder.fit_transform(chatterbox_temp['Title'])
chatterbox_temp['Status'] = label_encoder.fit_transform(chatterbox_temp['Status'])
chatterbox_temp['Designation'] = label_encoder.fit_transform(chatterbox_temp['Designation'])
chatterbox_temp['Religion'] = label_encoder.fit_transform(chatterbox_temp['Religion'])
chatterbox_temp['Date_Joined'] = label_encoder.fit_transform(chatterbox_temp['Date_Joined'])


train = chatterbox_temp[chatterbox_temp['Marital_Status'].notna()]

test = chatterbox_temp[chatterbox_temp['Marital_Status'].isna()]


"""Imputing 'Marital Status' using DecisionTree regression and cross validation."""

random_param = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [2, 3, 5, 10, 15, 20, 30, 50, 100, 200, None],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [1, 2, 3, 5, 10, 20, 30, 40, 50, 100],
    'max_features': [1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_leaf_nodes': [2, 5, 10, 20, 30, 50, 100, 1000, 2000, None]
}

clf = DecisionTreeClassifier()
rscv = RandomizedSearchCV(estimator=clf, n_iter=30000, param_distributions=random_param, scoring='accuracy', cv=5, n_jobs=-1, error_score='raise', verbose=3)
rscv.fit(train.loc[:, ~train.columns.isin(['Employee_No', 'Marital_Status'])], train['Marital_Status'])

# The best accuracy is given by the these hyperparameter values in this case,{'min_samples_split': 3, 'min_samples_leaf': 40, 'max_leaf_nodes': 10, 'max_features': 2, 'max_depth': None, 'criterion': 'entropy'}

dtc = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=3, max_leaf_nodes=1000, max_features=2, max_depth=30, criterion='entropy')
dtc.fit(train.loc[:, ~train.columns.isin(['Employee_No', 'Marital_Status'])], train['Marital_Status'])
predicted = dtc.predict(test.loc[:, ~test.columns.isin(['Employee_No', 'Marital_Status'])])

test['Marital_Status'] = predicted

count = 0
for i, r in test.iterrows():
    series = chatterbox_temp.loc[chatterbox_temp['Employee_No'] == r['Employee_No'], 'Marital_Status']
    if series.size == 1:
        chatterbox_temp.loc[chatterbox_temp['Employee_No'] == r['Employee_No'], 'Marital_Status'] = r['Marital_Status']
        count = count + 1

chatterbox['Marital_Status'] = chatterbox_temp['Marital_Status']




#Since Resigned_Date is the last date of the Inactive_Date, inactive column often consits of nan values and '0000' and '\N'. So here date resigned will bw same date as inactive day.

chatterbox.loc[chatterbox['Status'] == 'Active', 'Inactive_Date'] = '\\N'
chatterbox.loc[(chatterbox['Status'] == 'Inactive') & ((chatterbox['Date_Resigned'] == '\\N')
                                                       | (chatterbox['Date_Resigned'] == '0000-00-00'))
               & ((chatterbox['Inactive_Date'] != '\\N')
                  | (chatterbox['Inactive_Date'] != '0000-00-00')), 'Date_Resigned'] = chatterbox['Inactive_Date']
chatterbox.loc[chatterbox['Status'] == 'Active', 'Date_Resigned'] = '\\N'

"""When trying to impute the Year_of_Birth, We have to see how the values are given. Is it has outliers or not. """

print(chatterbox.dtypes)
chatterbox.loc[chatterbox['Year_of_Birth'] == "'0000'", 'Year_of_Birth'] = '0'
chatterbox['Year_of_Birth'] =chatterbox['Year_of_Birth'].astype('int64')

chatterbox_temp = chatterbox.drop(['Employee_Code', 'Name', 'Religion_ID', 'Designation_ID', 'Date_Resigned', 'Inactive_Date', 'Reporting_emp_1', 'Reporting_emp_2'], axis=1)
chatterbox_temp.loc[chatterbox_temp['Year_of_Birth'] == "'0000'", 'Year_of_Birth'] = '0'
chatterbox_temp['Year_of_Birth'] =chatterbox_temp['Year_of_Birth'].astype('int64')

for index, row in chatterbox_temp.iterrows():
    date = row['Date_Joined'].split('/')
    date = datetime(int(date[2]), int(date[0]), int(date[1]))
    chatterbox_temp.at[index, 'Date_Joined_Days'] = int(date.timestamp() / 86400)

print(chatterbox_temp[chatterbox_temp["Year_of_Birth"]== 0])

chatterbox_temp['Employment_Category'] = label_encoder.fit_transform(chatterbox_temp['Employment_Category'])
chatterbox_temp['Employment_Type'] = label_encoder.fit_transform(chatterbox_temp['Employment_Type'])
chatterbox_temp['Gender'] = label_encoder.fit_transform(chatterbox_temp['Gender'])
chatterbox_temp['Title'] = label_encoder.fit_transform(chatterbox_temp['Title'])
chatterbox_temp['Status'] = label_encoder.fit_transform(chatterbox_temp['Status'])
chatterbox_temp['Designation'] = label_encoder.fit_transform(chatterbox_temp['Designation'])
chatterbox_temp['Religion'] = label_encoder.fit_transform(chatterbox_temp['Religion'])
chatterbox_temp['Date_Joined'] = label_encoder.fit_transform(chatterbox_temp['Date_Joined'])
chatterbox_temp['Marital_Status'] = label_encoder.fit_transform(chatterbox_temp['Marital_Status'])

train = chatterbox_temp[chatterbox_temp['Year_of_Birth'] != 0]

test = chatterbox_temp[chatterbox_temp['Year_of_Birth'] == 0]

random_param = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'max_depth': [2, 3, 5, 10, 15, 20, 30, 50, 100, 200, None],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [1, 2, 3, 5, 10, 20, 30, 40, 50, 100],
    'max_features': [1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_leaf_nodes': [2, 5, 10, 20, 30, 50, 100, 1000, 2000, None]
}

reg = DecisionTreeRegressor()
rscv = RandomizedSearchCV(estimator=reg, n_iter=10000, param_distributions=random_param, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1, error_score='raise', verbose=3)
rscv.fit(train.loc[:, ~train.columns.isin(['Employee_No', 'Year_of_Birth'])], train['Year_of_Birth'])

#Here we have to use this parameters to get minimum error according to cross validation that we did in here.
#{'min_samples_split': 3, 'min_samples_leaf': 40, 'max_leaf_nodes': None, 'max_features': 5, 'max_depth': 3, 'criterion': 'friedman_mse'}

dtc = DecisionTreeRegressor(min_samples_split=3, min_samples_leaf=40, max_leaf_nodes=None, max_features=5, max_depth=3, criterion='friedman_mse')
dtc.fit(train.loc[:, ~train.columns.isin(['Employee_No', 'Year_of_Birth'])], train['Year_of_Birth'])
predicted = dtc.predict(test.loc[:, ~test.columns.isin(['Employee_No', 'Year_of_Birth'])])
test['Year_of_Birth'] = np.round(predicted)


count = 0
for i, r in test.iterrows():
    series = chatterbox_temp.loc[chatterbox_temp['Employee_No'] == r['Employee_No'], 'Marital_Status']
    if series.size == 1:
        chatterbox_temp.loc[chatterbox_temp['Employee_No'] == r['Employee_No'], 'Marital_Status'] = r['Marital_Status']
        count = count + 1
chatterbox['Year_of_Birth'] = chatterbox_temp['Year_of_Birth']
chatterbox.isna().sum()
# Writing the output dataframe to employee_preprocess_200638P.csv file
chatterbox.to_csv('employee_preprocess_200638P.csv')