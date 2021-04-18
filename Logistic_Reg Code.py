#Importing relevant libraries for data manipulation 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
import seaborn as sns

#Import dataset (Jan to April)
df=pd.read_csv(r'C:\Users\Zerxes\Desktop\FanCode Assesment\Train.csv')


##Changing date variables from continour into categiorical variables 
df['formatted_date'] = pd.to_datetime(df['Date no'])
df['day_of_year'] = df.formatted_date.apply(lambda x: x.dayofyear)
df['week_of_year'] = df.formatted_date.apply(lambda x: x.weekofyear)
df['dayOfWeek'] = pd.to_datetime(df['Date no']).apply(lambda x: x.weekday())
df['dayOfWeek_2'] = df['formatted_date'].dt.day_name()
df['Weekend_Y_N'] = df['dayOfWeek_2'].apply(lambda x: '0' if x in (['Saturday','Sunday']) else '1') # 0 for weekends
df = df.drop(['Date no','formatted_date','dayOfWeek_2'], axis=1)

# Create arrays for the features and the response variable
y = df['Distinct Users']
X = df.drop('Distinct Users', axis=1)

# Create varibale reg to be called in ML fucntion 
reg = LinearRegression()

# Split into training and test set (last 20% into testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)

# Fit the classifier to the data
reg.fit(X_train, y_train)

pred = reg.predict(X_test)

test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))

test_set_r2 = r2_score(y_test, pred)

print(test_set_rmse)
print(test_set_r2)
--984293.9372961122
---0.3689519550406337