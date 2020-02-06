import math
import quandl, datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_decomposition, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import *
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


style.use('ggplot')
# Reading the dat
quandl.ApiConfig.api_key = "V9hyFiFbPqE_sG_AupNi"
df = quandl.get('WIKI/GOOGL')

# Taking only important features
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]

# Adding features, i.e feature engineering
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df['h-l'] = (df['Adj. High'] - df['Adj. Low'])

df2 = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume', 'h-l']]
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'

# Replacing not avaliable(NA's) with -99999
df.fillna(-99999, inplace=True)
# Making next 10 predi
forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)

# Defining our features and labels
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[:-forecast_out:]
y = np.array(df['label'])
print(len(X), len(y))

# dividing data into test and train
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Training linearRegression model on our data
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
# saving the model
with open("linearregression.pickle", "wb") as f:
    pickle.dump(clf, f)

pickle_in = open("linearregression.pickle", "rb")
clf = pickle.load(pickle_in)


# Testing linearRegression model
accuracy = clf.score(X_test, y_test)
print(accuracy)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df["Forecast"] = np.nan

last_data = df.iloc[-1].name
last_unix = last_data.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df["Forecast"].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()













