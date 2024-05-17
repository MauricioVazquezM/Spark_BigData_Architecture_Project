## LIBRERIAS: 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import math
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import yfinance as yf
import time
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, window
from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')

# Initialize the figure
fig = plt.figure(figsize=(12, 16))


##### TIME SERIES FORESCASTING #####
# Create the large scatter plot at the top
symbol = 'AAPL'
data = yf.download(symbol, period="1h", interval="1m")
aux=[]
for i in data['Close']:
    aux.append(i)
df = pd.DataFrame(aux)
df['Close'] = df[0]

model =  SARIMAX(df["Close"], order=(1,1,1), seasonal_order=(1,1,1,5))
results = model.fit()

forecast = results.get_prediction(start=60, end=60+20)
mean_forecast = forecast.predicted_mean
confidence_intervals_1 = forecast.conf_int(alpha=0.01)
confidence_intervals_5 = forecast.conf_int(alpha=0.05)

empty5 = pd.DataFrame()
empty5["Close"] = df["Close"]
empty5["lower Close"] = empty5["Close"] 
empty5["upper Close"] = empty5["Close"] 

empty1 = pd.DataFrame()
empty1["Close"] = df["Close"]
empty1["lower Close"] = empty1["Close"] 
empty1["upper Close"] = empty1["Close"] 

confidence_intervals_5["Close"] = mean_forecast
confidence_intervals_5 = pd.concat([empty5, confidence_intervals_5])

confidence_intervals_1["Close"] = mean_forecast
confidence_intervals_1 = pd.concat([empty1, confidence_intervals_1])

plt.rcParams['figure.figsize'] = [16, 5]

ax0 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.spines['left'].set_visible(False)
    
ax0.plot(range(60+21), confidence_intervals_5["Close"], color='purple', label='forecast')
ax0.fill_between(range(60+21), confidence_intervals_1["lower Close"], confidence_intervals_1["upper Close"], color='cyan', alpha=0.8)
ax0.fill_between(range(60+21), confidence_intervals_5["lower Close"], confidence_intervals_5["upper Close"], color='pink', alpha=0.9)
ax0.bar(range(60+21), confidence_intervals_5["Close"], width=0.3, color='pink', alpha=0.1)

ax0.set_ylabel("Precio Accion: "+str(symbol))
 
maximo=max(confidence_intervals_5['Close'].max(),confidence_intervals_5['upper Close'].max())

minimo=min(confidence_intervals_5['Close'].min(),confidence_intervals_5['lower Close'].min())

ax0.set_ylim(minimo-.25,maximo+.25)

ax0.set_title('Time Series Forecasting')


##### MATRICES #####
# Seed for reproducibility
np.random.seed(0)

# Creating a DataFrame with 10 variables and 100 observations
data = {
    'Var1': np.random.randn(100),
    'Var2': np.random.randn(100),
    'Var3': np.random.randn(100),
    'Var4': np.random.randn(100),
    'Var5': np.random.randn(100)
}

df = pd.DataFrame(data)

# Compute the initial correlation matrix
correlation_matrix = df.corr()

# Set up the plotting figure and axes
ax1 = plt.subplot2grid((3, 2), (1, 0))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',cbar=False, ax=ax1)
ax1.set_title('Real-time Correlation Matrix')

# Set up the plotting figure and axes
ax2 = plt.subplot2grid((3, 2), (1, 1))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',cbar=False, ax=ax2)
ax2.set_title('Real-time Correlation Matrix')

# Set up the plotting figure and axes
ax3 = plt.subplot2grid((3, 2), (2, 0))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',cbar=False, ax=ax3)
ax3.set_title('Real-time Correlation Matrix')

# Set up the plotting figure and axes
ax4 = plt.subplot2grid((3, 2), (2, 1))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',cbar=False, ax=ax4)
ax4.set_title('Real-time Correlation Matrix')

# Function to update the data in the heatmap
def update(frame):
    global df, correlation_matrix
    # Simulate incoming data: appending new row of random data
    new_row = np.random.randn(1, 5)  # Corrected to generate new data for all 10 variables
    new_df = pd.DataFrame(new_row, columns=df.columns)
    df = pd.concat([df, new_df], ignore_index=True)
    
    # Limit the size of df to the most recent 100 rows
    df = df.tail(100)
    
    # Recalculate the correlation matrix
    new_correlation_matrix = df.corr()
    
    # Efficiently update the existing heatmap data
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            ax1.texts[i * len(correlation_matrix.columns) + j].set_text(f"{new_correlation_matrix.iloc[i, j]:.2f}")
            ax2.texts[i * len(correlation_matrix.columns) + j].set_text(f"{new_correlation_matrix.iloc[i, j]:.2f}")
            ax3.texts[i * len(correlation_matrix.columns) + j].set_text(f"{new_correlation_matrix.iloc[i, j]:.2f}")
            ax4.texts[i * len(correlation_matrix.columns) + j].set_text(f"{new_correlation_matrix.iloc[i, j]:.2f}")


# Funcion animacion
ani = FuncAnimation(fig, update, frames=np.arange(0, 200), blit=False, interval=1000, repeat=True) 

# Showing Dashboard
plt.tight_layout(pad=4)
plt.show()