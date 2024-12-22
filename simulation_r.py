#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 16:00:00 2024

@author: hakan
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from mplfinance.original_flavor import candlestick_ohlc
import talib
import mplfinance as mpf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import load_img
import tensorflow as tf
import shutil

def find_time_index(time,df):

        df=pd.DataFrame(df)
        index = df.index.get_loc(time)
        print(index)
        return index
    
gpus = tf.config.list_physical_devices('GPU')
# if gpus: 
#     for gpu in gpus:
#           tf.config.experimental.set_memory_growth(gpu, True)
if gpus: 
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=7492)]
    )

logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

dd = '/home/hakan/Desktop/Edrees/EURUSD_M15-test.csv'
firstDate="2024-03-01 20:45:00"
lastDate="2024-10-31 20:45:00"

#this part creates images without labels. Because we predict the labels using our cnn model. 

data = pd.read_csv(dd, delimiter=',', index_col='Time', parse_dates=True)
data['SMA'] = talib.SMA(data['Close'], timeperiod=20)
initialTime_index=find_time_index(firstDate,data)
finalTime_index=find_time_index(lastDate,data)
data=data[initialTime_index:finalTime_index]

# output_dir = "test_for_signal"
# shutil.rmtree(output_dir,ignore_errors=True)
# os.makedirs(output_dir, exist_ok=True)
# window_size=5
# shift_size=2
# for i in range(0, len(data) - window_size,shift_size):
#     window = data.iloc[i:i+window_size]
#     save_path = os.path.join(output_dir, f"{window.iloc[-1].name}.png")
#     ap = [mpf.make_addplot(window['SMA'], color='blue', secondary_y=False)]
#     mpf.plot(window, type='candle', style='yahoo', addplot=ap, volume=True, axisoff=True, ylabel='',
#  savefig=save_path)
#     plt.close()

# Create DataFrame
data = pd.read_csv(dd, delimiter=',', parse_dates=True)
data=data[initialTime_index:finalTime_index]
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Time'])
df['Date'] = df['Date'].map(mdates.date2num)  # Convert dates to matplotlib format

# Create subplots and plot candlestick chart
fig, ax = plt.subplots(figsize=(14, 7))
candlestick_ohlc(ax, df[['Date', 'Open', 'High', 'Low', 'Close']].values, width=0.01, colorup='green', colordown='red')


# This part makes predictions 
from tensorflow.keras.utils import img_to_array

dataset_path="test_for_signal"
X=[]
for name in os.listdir(dataset_path):
    image1 = load_img(dataset_path  + '/' + name, color_mode = 'rgb', interpolation="bilinear",target_size = (150, 150) )  # MODEL 2 & MODEL 3 (analyzing each image as a whole)
    image1 = img_to_array(image1)
    image1 = image1 / 255
    X.append(image1)
X=np.array(X) 
model=load_model("chart_classification_model_W5.h5")
predictions = model.predict(X)

image_names=os.listdir(dataset_path)
indicator_xcoordinates=[]
indicator_trends=[]
for idx,i in enumerate(predictions):
    if i>=0.5:
        indicator_xcoordinates.append(os.path.splitext(image_names[idx])[0])
        indicator_trends.append("U")
    else:
        indicator_xcoordinates.append(os.path.splitext(image_names[idx])[0])
        indicator_trends.append("D")

## remove consecutive the same signals. That is the list will be [up,down,up,down,up, down...] and so on
signal_x = [indicator_xcoordinates[0]]  # Start with the first element
signal_label = [indicator_trends[0]] 
for i in range(1, len(indicator_trends)):
    if indicator_trends[i] != indicator_trends[i - 1]:
        signal_x.append(indicator_xcoordinates[i])
        signal_label.append(indicator_trends[i])
        
indicator_xcoordinates=signal_x
indicator_trends=signal_label  
   
# Add annotations for up/down labels
for time, label in zip(indicator_xcoordinates, indicator_trends):
    # Get the row data for that specific time
    if time in df['Time'].values:
        result = df.isin([time])
        locations = result.stack()[result.stack()] 
        row = locations.index[0][0]
        row=df.loc[row]
        timestamp = mdates.date2num(pd.to_datetime(time))
        # Position label above the high price with a small offset
        if label=='D':
            y_position = row['High'] + 0.00022  # Adjust offset as needed
        else:
            y_position = row['Low'] - 0.00032  # Adjust offset as needed
        ax.annotate(label,
                    xy=(timestamp, y_position),
                    xytext=(0, 2),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', 
                             fc='yellow', 
                             alpha=0.5)
                    )
    else:
            print(f"Time {time} not found in data")
            print(time)
# Formatting the plot
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d,%H:%M'))
plt.title('Up/Down Trend Signals for EUR/USD ')
plt.xticks(rotation=60)
plt.xlabel('Date-Time')
plt.ylabel('Price')
plt.grid(True)
plt.tight_layout()
plt.show()


initial_amount_usd = 1000  # Initial amount in USD
current_amount_usd = initial_amount_usd
amount_in_euros = 0  # Amount of euros after buying
number_changes=0
for i in range(len(indicator_xcoordinates)):
    time=indicator_xcoordinates[i]
    if time in df['Time'].values:
        result = df.isin([time])
        locations = result.stack()[result.stack()] 
        row = locations.index[0][0]
        row=df.loc[row]
    if indicator_trends[i] == 'U' and current_amount_usd > 0:  # Buy signal   
        amount_in_euros = current_amount_usd / row['Open']
        number_changes+=1
        current_amount_usd = 0  # All money converted to euros
        print(f"Bought at {time} at price {row['Open']}, amount in euros: {amount_in_euros:.2f}")
        
    elif indicator_trends[i] == 'D' and amount_in_euros > 0:  # Sell signal
        current_amount_usd = amount_in_euros * row['Open']
        amount_in_euros = 0  # Euros converted back to USD
        print(f"Sold at {time} at price {row['Open']}, amount in USD: {current_amount_usd:.2f}")
        number_changes+=1

# Final amount
if amount_in_euros > 0:  # Convert any remaining euros to USD at the last close price
    print(f"\nFinal amount in EUR: {amount_in_euros:.2f}")
    current_amount_usd=amount_in_euros*data['Open'][finalTime_index-1]
    print(f"\nFinal amount in Dollar: {current_amount_usd:.2f}")
else:
    print(f"\nFinal amount in Dollar: {current_amount_usd:.2f}")
    
print(f"\nTotal number of buy/sell: {number_changes}")