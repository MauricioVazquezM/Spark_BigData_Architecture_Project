# LIBRERIAS: 
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
matplotlib.use('Agg')
import os

print("BUENOS DIAS WILMER")
## Layout set up
st.set_page_config(page_title="Financial Analysis Dashboard", page_icon="📈", layout="wide")


## Page tittle
st.title("FINANCIAL ANALYSIS DASHBOARD")



# Desplegando multiselectbox para seleccionar caso HIT o NO HIT
financial_services = ['BRK-B', 'JPM', 'BAC', 'WFC', 'GS']
consumer_staples = ['PG', 'PEP', 'KO', 'COST', 'WMT']
consumer_discretionary = ['AMZN', 'TSLA', 'MCD', 'HD', 'LOW']
communication_services = ['META', 'GOOGL', 'GOOG', 'NFLX', 'AAPL']
all_symbols = financial_services + consumer_staples + consumer_discretionary+communication_services

### WIDGET: ACTIONS (LISTA PARA SELECCION MULTIPLE)
sym ='AAPL'


### WIDGET: BOTON START (BOTON)


# Initialize the figure
#fig, axs = plt.subplots(3, 2, figsize=(12, 16))

##### MATRICES #####

# Inicializar SparkSession
spark = SparkSession.builder.appName("YahooFinanceStreaming").getOrCreate()

# Crear un esquema para el DataFrame de PySpark
schema = StructType([
    StructField("symbol", StringType(), True),
    StructField("close", FloatType(), True),
    StructField("timestamp", TimestampType(), True)
])

# Crear un DataFrame vacío con el esquema definido
df_spark = spark.createDataFrame(spark.sparkContext.emptyRDD(), schema)
df2 = pd.DataFrame(columns=['AAPL', 'AMZN', 'BAC', 'BRK-B', 'COST', 'GOOG', 'GOOGL', 'GS', 'HD', 'JPM', 'KO', 'LOW', 'MCD', 'META', 'NFLX', 'PEP', 'PG', 'TSLA', 'WFC', 'WMT'])

# Create placeholders for each plot
forecast_placeholder = st.empty()
col1, col2 = st.columns(2)
financial_placeholder = col1.empty()
staples_placeholder = col1.empty()
discretionary_placeholder = col2.empty()
communication_placeholder = col2.empty()
current_time=datetime.now()
for i in range(0, 5):
    for symbol in all_symbols:
        data = yf.download(symbol, interval='1m', period='1d')
        if not data.empty:
            close_price = float(data['Close'].tail(1)[0])
            datetime_obj = data.index[-1].to_pydatetime()

            # Crear un nuevo DataFrame con los datos obtenidos
            new_data = spark.createDataFrame([(symbol, close_price, datetime_obj)], schema)
            # Concatenar el nuevo DataFrame con el DataFrame existente
            df_spark = df_spark.union(new_data)
            ##show last 20
    #print(df_spark.toPandas().tail(20))

    time.sleep(10)
    if i%2==0:
        #symbol_aux = 'AAPL'
        data_aux = yf.download(sym, period="1d", interval="1m").tail(60)
        aux=[]
        for i in data_aux['Close']:
            aux.append(i)
        df3 = pd.DataFrame(aux)
        df3['Close'] = df3[0]

        with forecast_placeholder.container():

            model =  SARIMAX(df3["Close"], order=(1,1,1), seasonal_order=(1,1,1,5))
            results = model.fit()
            # results.summary()

            forecast = results.get_prediction(start=60, end=60+20)
            mean_forecast = forecast.predicted_mean
            confidence_intervals_1 = forecast.conf_int(alpha=0.01)
            confidence_intervals_5 = forecast.conf_int(alpha=0.05)

            empty5 = pd.DataFrame()
            empty5["Close"] = df3["Close"]
            empty5["lower Close"] = empty5["Close"] 
            empty5["upper Close"] = empty5["Close"] 

            empty1 = pd.DataFrame()
            empty1["Close"] = df3["Close"]
            empty1["lower Close"] = empty1["Close"] 
            empty1["upper Close"] = empty1["Close"] 

            confidence_intervals_5["Close"] = mean_forecast
            confidence_intervals_5 = pd.concat([empty5, confidence_intervals_5])

            confidence_intervals_1["Close"] = mean_forecast
            confidence_intervals_1 = pd.concat([empty1, confidence_intervals_1])

            min_close = confidence_intervals_1['Close'].min()
            max_close = confidence_intervals_1['Close'].max()
            mean = confidence_intervals_1['Close'].mean()

            plt.rcParams['figure.figsize'] = [16, 5]
            fig, ax = plt.subplots()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
                
            ax.plot(range(60+21), confidence_intervals_5["Close"], color='purple', label='forecast')
            ax.fill_between(range(60+21), confidence_intervals_1["lower Close"], confidence_intervals_1["upper Close"], color='cyan', alpha=0.8)
            ax.fill_between(range(60+21), confidence_intervals_5["lower Close"], confidence_intervals_5["upper Close"], color='pink', alpha=0.9)
            ax.bar(range(60+21), confidence_intervals_5["Close"], width=0.3, color='pink', alpha=0.1)
            ax.set_ylabel("Precio Accion: "+str(symbol))

            # Add horizontal lines for min and max values
            plt.axhline(y=min_close, color='blue', linestyle='--', label='Min Close')
            plt.axhline(y=max_close, color='red', linestyle='--', label='Max Close')
            plt.axhline(y=mean, color='green', linestyle='--', label='Mean Close')

            maximo=max(confidence_intervals_5['Close'].max(),confidence_intervals_5['upper Close'].max())

            minimo=min(confidence_intervals_5['Close'].min(),confidence_intervals_5['lower Close'].min())

            plt.ylim(minimo-.25,maximo+.25)

            plt.legend()
            
            st.pyplot(fig)


    if i >= 1:
        df_spark.createOrReplaceTempView("stock_data")

        # Consulta para calcular la diferencia entre los últimos dos registros en df para todas las acciones
        result_delta = spark.sql("""
            SELECT
                latest.symbol,
                latest.close - previous.close AS delta
            FROM (
                SELECT
                    symbol,
                    close,
                    timestamp,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS rn
                FROM stock_data
            ) latest
            JOIN (
                SELECT
                    symbol,
                    close,
                    timestamp,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) AS rn
                FROM stock_data
            ) previous
            ON latest.symbol = previous.symbol AND latest.rn = 1 AND previous.rn = 2
        """)
        res = result_delta.toPandas()
        new_columns = res['symbol'].tolist()
        new_data = res.drop(columns=['symbol']).T
        new_data.columns = new_columns
        df2 = pd.concat([df2, new_data], ignore_index=True)
        df2_financial_services = df2[['BRK-B', 'JPM', 'BAC', 'WFC', 'GS']].tail(100)
        df2_consumer_staples = df2[['PG', 'PEP', 'KO', 'COST', 'WMT']].tail(100)
        df2_consumer_discretionary = df2[['AMZN', 'TSLA', 'MCD', 'HD', 'LOW']].tail(100)
        df2_communication_services = df2[['META', 'GOOGL', 'GOOG', 'NFLX', 'AAPL']].tail(100)

        #col1, col2 = st.columns(2)

        # Update plots in the placeholders
        with financial_placeholder.container():
            correlation_matrix_financial_services = df2_financial_services.corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix_financial_services, annot=True, fmt=".2f", cmap='coolwarm')
            plt.title('Real-time Correlation Matrix Financial Services')
            st.pyplot(plt)

        with staples_placeholder.container():
            correlation_matrix_consumer_staples = df2_consumer_staples.corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix_consumer_staples, annot=True, fmt=".2f", cmap='coolwarm')
            plt.title('Real-time Correlation Matrix Consumer Staples')
            st.pyplot(plt)

        with discretionary_placeholder.container():
            correlation_matrix_consumer_discretionary = df2_consumer_discretionary.corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix_consumer_discretionary, annot=True, fmt=".2f", cmap='coolwarm')
            plt.title('Real-time Correlation Matrix Consumer Discretionary')
            st.pyplot(plt)

        with communication_placeholder.container():
            correlation_matrix_communication_services = df2_communication_services.corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix_communication_services, annot=True, fmt=".2f", cmap='coolwarm')
            plt.title('Real-time Correlation Matrix Communication Services')
            st.pyplot(plt)
latest_time=datetime.now()
print(latest_time-current_time)
