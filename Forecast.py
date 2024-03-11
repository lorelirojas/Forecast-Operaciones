#Librerias
import warnings  #evitar warnings
import pandas as pd  #trabajar tablas y estructuras de datos
from datetime import datetime
from pandas import DataFrame
import numpy as np #vectores y matrices multidimensiones y operaciones complejas
import matplotlib.pyplot as plt  #para trabajar graficos
from openpyxl import Workbook #en caso de ser necesario instalar la libreria openpyxl y workbook para excel
import statsmodels.api as sm  #explorar modelos estadisticos
from statsmodels.tsa.seasonal import seasonal_decompose #descomposición de la serie
from sklearn.metrics import mean_squared_error, mean_absolute_error #biblioteca de aprendizaje automatico
from math import sqrt #operaciones matematicas
import pymannkendall as mk #Prueba para comprobar si hay tendencia
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta #pip install python-dateutil
# Modelos
from statsmodels.tsa.ar_model import AutoReg #Autoregesión (AR)
from statsmodels.tsa.arima_model import ARMA # Autoregresión con media movil (ARMA)
import pmdarima as pm # Autoregresión Integrado con Media movil (ARIMA) y Autoregresión Integrado con Media movil Estacional (SARIMA)
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # Modelo Exponencial Simple
from statsmodels.tsa.api import ExponentialSmoothing #Modelo Holt y Modelo Holt-Winters
# Widgets
from google.colab import widgets


df=pd.read_excel("DatosForecast.xlsx")


#Convertir la fecha en un formato pandas
df.Timestamp = pd.to_datetime (df.iloc[:, 0], format = "%Y/%m/%d") #indexa la fecha, le da formato
df.index = df.Timestamp
#Eliminar la columna extra de Fecha
df=df.drop(["Fecha"],axis=1)
#Convertir lo valores negativos en 0
df[df<0]=0
df.info()

import ipywidgets as widgets

# Visualizar la Demanda

# Create a list to store the output widgets for each tab
output_widgets = [widgets.Output() for _ in range(len(df.columns))]

# Create tabs with labels and corresponding output widgets
tabs = widgets.Tab(children=output_widgets)
for i in range(len(df.columns)):
    tabs.set_title(i, "SKU {}".format(i + 1))

# Display the tabs
display(tabs)

# Loop through columns and plot in corresponding output widget
for i in range(len(df.columns)):
    with output_widgets[i]:
        plt.plot(df.iloc[:, i], label="Demanda del SKU {}.".format(i + 1))
        plt.legend(loc='best')
        plt.show()
