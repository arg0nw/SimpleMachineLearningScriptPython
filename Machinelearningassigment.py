
# coding: utf-8

# In[10]:


import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as pt
from matplotlib import style

style.use('ggplot')

#getting data from the quandl
quandl.ApiConfig.api_key = 'hDgr2JYTKBDqUNbsArkM'
data_field= quandl.get('SSE/SSU', start_date='2010-12-06', end_date='2018-06-01')

#extracting data from the dataset
data_field = data_field [['High','Low','Last','Volume']]

#defining % of High and low value ratio
data_field['High_Low_%']=(data_field['High'] - data_field['Low']) / data_field['Low'] * 100

#setting up data set
data_field = data_field[['High_Low_%', 'Last', 'Volume']]

#defining forecasting column
forecasting_col='Last'

#removing null values and replacing those with -99999
data_field.fillna(-99999, inplace=True)

#defining forecasting output
forecasting_output = int(math.ceil(0.01*len(data_field)))

#defining label in dataset
data_field['label'] = data_field[forecasting_col].shift(-forecasting_output)

#set up x
x = np.array(data_field.drop(['label'], 1))
x = preprocessing.scale(x)
x = x[:-forecasting_output]
x_toPredict = x[-forecasting_output:]

#set up y
data_field.dropna(inplace=True)
y = np.array(data_field['label'])

#shuffuling and spliting the dataset
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2)

#using LinearRegression
clf= LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

#predic using x data
forecast_set = clf.predict(x_toPredict)

print(forecast_set, accuracy, forecasting_output)

data_field['Forecast'] = np.nan

#set up dates
last_date = data_field.iloc[-1].name
last_unix = last_date.timestamp()
one_day=86400
next_unix = last_unix+one_day

#iteration each forecast
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    data_field.loc[next_date] = [np.nan for _ in range(len(data_field.columns)-1)] + [i]
    
#plotting the prediction    
data_field['Last'].plot()
data_field['Forecast'].plot()
pt.legend(loc=1)
pt.xlabel('Date')
pt.ylabel('Price')
pt.rcParams['figure.figsize'] = (50,20)

pt.show()



