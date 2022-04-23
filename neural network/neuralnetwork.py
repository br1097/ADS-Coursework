import pandas as pd
from pandas import read_csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

dataframe = pd.read_csv('NNtesting2.csv', header=0)
dataset = dataframe.values
dataset = dataset.astype('float')
#print(dataset[:10])
# plt.plot(np.arange(0,400), dataset, 'b-')
# plt.show()

scaler = MinMaxScaler(feature_range=(0,1))
dataset_for_training = scaler.fit_transform(dataset)

column = list(dataframe)[0]
air_temp_column = dataframe[column].astype('float')
#print(columns)

x_train = []
y_train = []

look_back = 20
look_forward = 1

for i in range(look_back, len(dataset_for_training) - look_forward +1):

   x_train.append(dataset_for_training[i - look_back:i, 0:dataset.shape[1]])
   y_train.append(dataset_for_training[i + look_forward - 1:i + look_forward, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

model = Sequential()

model.add(LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]), activation='sigmoid', return_sequences=True))
model.add(LSTM(32, activation='sigmoid',return_sequences=False))
model.add(Dense(y_train.shape[1]))

opt = tf.keras.optimizers.Adam(learning_rate=1e-2)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])

model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.1, verbose=1)

n_future = 50
prediction = model.predict(x_train[-n_future:])

prediction_copies = np.repeat(prediction, dataset.shape[1], axis=-1)
y_prediction_future = scaler.inverse_transform(prediction_copies)[:,0]
#print(y_prediction_future)

#pd.DataFrame(x_test_predict).to_csv("predicted_data.csv")

fig, ax = plt.subplots()
#ax.plot(np.arange(len(dataset)), air_temp_column, 'b-') #actual
ax.plot( np.arange(0,len(dataset)-len(y_prediction_future)), air_temp_column[:-len(y_prediction_future)], 'b-' ) #actual
ax.plot(np.arange(len(dataset)-len(y_prediction_future),len(dataset)), y_prediction_future, 'r-') #forecast
#ax.plot(np.arange(len(dataset)-len(x_test_predict),len(dataset)-len(x_predict)),x_test_predict[0:-len(x_predict)], 'c-')
ax.set_ylabel('Air temperature (K)')
ax.legend(["Real data", "Prediction"], loc="upper left")
plt.grid()
plt.show()