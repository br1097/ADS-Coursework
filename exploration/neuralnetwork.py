import pandas as pd
from pandas import read_csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

dataframe = pd.read_csv('temperature.csv', header=0, usecols=["Air temperature"])
dataset = dataframe.values
dataset = dataset.astype('float')
#print(dataset)

scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

look_back = 1
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
#train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
x_train = dataset[0:train_size]
y_train = dataset[0:train_size]
x_test = dataset[train_size:len(dataset)]
y_test = dataset[train_size + 1:len(dataset)]
x_train = np.reshape(x_train, (x_train.shape[0], look_back, 1))
y_train = np.reshape(y_train, (y_train.shape[0], look_back, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))

predict_size = 20
x_test_predict, y_test_predict = x_test[0:int(x_test.shape[0]-predict_size),0], y_test[0:int(y_test.shape[0])-predict_size,0]
x_predict,y_predict = x_test[-predict_size:x_test.shape[0],0], y_test[-predict_size:y_test.shape[0],0]
#print(len(x_test_predict))

model = Sequential()

model.add(LSTM(500, input_shape=(x_train.shape[1], x_train.shape[2]), activation='relu', return_sequences=True))
model.add(LSTM(100, activation='relu'))

model.add(Dense(1, activation="relu"))

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
model.fit(x_train, y_train, epochs=50, batch_size=200)

look_forward = 20
day = 0
prices = np.zeros((x_test_predict.shape[0]+look_forward,1))
prices[0:x_test_predict.shape[0],0] = x_test_predict
while day<=look_forward:
   x_test_predict = np.reshape(x_test_predict,(x_test_predict.shape[0],look_back,1))
   pred=model.predict(x_test_predict)
   x_test_predict = np.reshape(x_test_predict, (x_test_predict.shape[0], 1))
   prices[0:x_test.shape[0]+day, 0] = pred[-1]
   x_test_predict_temporary = np.zeros((x_test_predict.shape[0]+1,1))
   x_test_predict_temporary[0:x_test_predict.shape[0],0] = x_test_predict[:,0]
   x_test_predict_temporary[-1] = pred[-1]
   x_test_predict = x_test_predict_temporary
   day += 1

final_predict = x_test_predict[-look_forward:len(x_test_predict)]

#pd.DataFrame(x_test_predict).to_csv("predicted_data.csv")

fig, ax = plt.subplots()
ax.plot(np.arange(len(dataset)), dataset[0:len(dataset)], 'b-')
ax.plot(np.arange(len(dataset)-len(x_predict),len(dataset)), final_predict, 'c-')
ax.plot(np.arange(len(dataset)-len(x_test_predict),len(dataset)-len(x_predict)),x_test_predict[0:-len(x_predict)], 'r-')
#ax.set_ylabel('Price (ppl)')
#plt.xlabel('Time')
ax.legend(["Actual", "Forecast", "Prediction"], loc="upper left")
plt.grid()
plt.show()