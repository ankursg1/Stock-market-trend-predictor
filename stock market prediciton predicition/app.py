import numpy as np
import pandas as pd
from pandas_datareader import data as web
import yfinance as yfin
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import load_model

import streamlit as st

# Defining the start and the end date for our data

st.title("Stock Trend Prediction")
user_input = st.text_input("Enter Stock Ticker" , 'AAPL')  # taking user input


yfin.pdr_override()
df = web.DataReader(user_input,'2000-01-01','2023-02-9' )


# Describing the Data

st.subheader('Data from 2000-2023')
st.write(df.describe())



# Visualizations

st.subheader('Closing price VS Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


# Visualizations with 100MA

st.subheader('Closing price VS Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)



# Visualizations with 200MA

st.subheader('Closing price VS Time chart with 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100 , 'r')
plt.plot(ma200 , 'b')
plt.plot(df.Close, 'g')
st.pyplot(fig)






# splitting data for training and testing



# 70% data for training and 30% data for testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) 
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


# SCALING UP THE DATA
from sklearn.preprocessing import MinMaxScaler
# feature_range=(0,1)  ---> means all the values of the column of the scaler will be scaled down
#  between 1 and 0
scaler  = MinMaxScaler(feature_range=(0,1))


# FIT TRANSFORMING THE DATA
data_training_array = scaler.fit_transform(data_training)




# SPLITTING DATA INTO x_train , y_train

# x_train = []
# y_train = []

# # appending our stock prices in the training data

# # for i in range (100,data_training_array.shape[0]): ----> the value of 101 index of (close) will depend on
# #  100 previous values

# for i in range (100,data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100:i])
#     y_train.append(data_training_array[i,0])


# # now we have to convert it into numpy array so that we can pass it to the LSTM
# x_train , y_train = np.array(x_train),np.array(y_train)

# -----------------------------we have pertained the model---------------------------------------




# LOADING MY MODEL

model = load_model('keras_model.h5')


# testing part

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index = True)

# scaling again
input_data = scaler.fit_transform(final_df)



x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100 : i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test) , np.array(y_test)



# Predicting
y_predict = model.predict(x_test)


# scaling the output data

scaler = scaler.scale_
scale_factor = 1/scaler[0]

y_predict = y_predict*scale_factor
y_test = y_test*scale_factor



# Visualizing the prediction

st.subheader('Predictions vs original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test , 'b' , label = "Original Price")
plt.plot(y_predict , 'r' , label = "Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)