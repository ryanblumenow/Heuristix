from itertools import count
from os import curdir
import hydralit as hy
from numpy.core.fromnumeric import var
from sqlalchemy import null
import streamlit
import streamlit as st
import sys
from streamlit.web import cli as stcli
from PIL import Image
from functions import *
import streamlit.components.v1 as components
import pandas as pd
from st_clickable_images import clickable_images
import numpy as np
import statsmodels.api as sm
from numpy import empty, mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import seaborn as sns
from io import BytesIO
from statsmodels.formula.api import ols
# from streamlit.state.session_state import SessionState
import tkinter
import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('Agg')
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.tree import DecisionTreeRegressor, plot_tree
import sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
import time
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import dtale
from dtale.views import startup
from dtale.app import get_instance
import webbrowser
import dtale.global_state as global_state
import dtale.app as dtale_app
from matplotlib.pyplot import hist
from scipy import stats as stats
from bioinfokit.analys import stats
from streamlit_option_menu import option_menu
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from sklearn.preprocessing import LabelEncoder
from streamlit_quill import st_quill

#add an import to Hydralit
from hydralit import HydraHeadApp

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

@st.cache

def readdata():

    # Read data
    df = pd.read_csv('allrecordsohe.csv', low_memory=False)
    df2 = pd.read_csv('allrecords.csv', low_memory=False)
    branddf = pd.read_csv('Brandname encoding.csv', low_memory=False)

    # Check for empty data
    df.isnull().sum()
    df2.isnull().sum()

    # Remove NaN
    nr_samples_before = df.shape[0]
    df = df.fillna(0)
    print('Removed %s samples' % (nr_samples_before - df.shape[0]))
    nr_samples_before = df2.shape[0]
    df2 = df2.fillna(0)
    print('Removed %s samples' % (nr_samples_before - df2.shape[0]))

    # Drop irrelevant variables
    df.drop(['TD_ID', 'KRUX_ID', 'TAP_IT_ID', 'GOOGLE_CLIENT_ID'], axis=1, inplace=True)
    df2.drop(['TD_ID', 'KRUX_ID', 'TAP_IT_ID', 'GOOGLE_CLIENT_ID'], axis=1, inplace=True)

    # df = df.reset_index()
    # df2 = df2.reset_index()

    return df, df2, branddf

    ### End

    ### Enter code to test here

df, df2, branddf = readdata()

title = '<p style="font-family:sans-serif; color:red; font-size: 39px; text-align: center;"><b>Custom data/code testing environment</b></p>'
st.markdown(title, unsafe_allow_html=True)


def exponential_fit(x, a, b, c):
    return a*np.exp(-b*x) + c

x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([30, 50, 80, 160, 300, 580])
fitting_parameters, covariance = curve_fit(exponential_fit, x, y)
a, b, c = fitting_parameters

next_x = 6
next_y = exponential_fit(next_x, a, b, c)

plt.plot(y)
plt.plot(np.append(y, next_y), 'ro')
st.pyplot(plt)



# THIS ONE

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# print(datetime.fromisoformat('2014-03-05').timestamp())

x1 = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
# x = np.arange("2020-03","2020-04",dtype='datetime64[D]')
x2  = np.array(['2014-01-05', '2014-02-05', '2014-03-05', '2014-04-05', '2014-05-05', '2014-06-05'])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
# y = np.array([0.0, 0.8, 0.9, 1.1, 1.3, 1.5])

explvartype = st.selectbox("Is the explanatory variable a date field or a variable?", options=['Date', "Variable"])

if explvartype == "Date":

    x = x2

    a = []

    for i in x:
        counter = 0
        numericaldate = datetime.fromisoformat(i).timestamp()/1000000000
        a.append(numericaldate)
        counter += 1

    newx = np.array(a)
    print("newx starts here")
    print(newx)

elif explvartype == "Variable":

    newx = x1

# print(type(a[0]))
# z = np.polyfit(newx, y, 3)
try:
    z = np.polyfit(newx, y, 3)
except:
    try:
        z = np.polyfit(newx, y, 2)
    except:
        try:
            z = np.polyfit(newx, y, 1)
        except:
            st.write("No polynomial fits the parameters. Please check the data since even linear extrapolation is impossible.")

# st.write(z)

# create polynomial
p = np.poly1d(z)

# plot polynomial
xp = np.linspace(min(newx), max(newx), 100)
_ = plt.plot(newx, y, '.', xp, p(xp), '-')
plt.ylim(-2,2)
plt.xlim(min(newx), max(newx))
# plt.xscale
st.pyplot(plt)

# show roots
st.write(p.roots)
st.write("Prediction 1:")
st.write(p(datetime.fromisoformat('2014-10-05').timestamp()/1000000000))


# AND THIS ONE

import numpy as np
from scipy import interpolate

# x = np.arange(0,10)
# y = np.exp(-x/3.0)
x  = np.array(['2014-01-05', '2014-02-05', '2014-03-05', '2014-04-05', '2014-05-05', '2014-06-05'])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])

a = []

for i in x:
    counter = 0
    numericaldate = datetime.fromisoformat(i).timestamp()/1000000000
    a.append(numericaldate)
    counter += 1

newx = np.array(a)

print(newx)

st.write("Prediction 2:")
f = interpolate.interp1d(newx, y, fill_value='extrapolate', kind='cubic')

st.text(f(datetime.fromisoformat('2014-10-05').timestamp()/1000000000))





import numpy, scipy, matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings

# xData = numpy.array([19.1647, 18.0189, 16.9550, 15.7683, 14.7044, 13.6269, 12.6040, 11.4309, 10.2987, 9.23465, 8.18440, 7.89789, 7.62498, 7.36571, 7.01106, 6.71094, 6.46548, 6.27436, 6.16543, 6.05569, 5.91904, 5.78247, 5.53661, 4.85425, 4.29468, 3.74888, 3.16206, 2.58882, 1.93371, 1.52426, 1.14211, 0.719035, 0.377708, 0.0226971, -0.223181, -0.537231, -0.878491, -1.27484, -1.45266, -1.57583, -1.61717])
# yData = numpy.array([0.644557, 0.641059, 0.637555, 0.634059, 0.634135, 0.631825, 0.631899, 0.627209, 0.622516, 0.617818, 0.616103, 0.613736, 0.610175, 0.606613, 0.605445, 0.603676, 0.604887, 0.600127, 0.604909, 0.588207, 0.581056, 0.576292, 0.566761, 0.555472, 0.545367, 0.538842, 0.529336, 0.518635, 0.506747, 0.499018, 0.491885, 0.484754, 0.475230, 0.464514, 0.454387, 0.444861, 0.437128, 0.415076, 0.401363, 0.390034, 0.378698])
xData  = np.array(['2014-01-05', '2014-02-05', '2014-03-05', '2014-04-05', '2014-05-05', '2014-06-05'])
yData = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])

xData1 = []

for i in xData:
    counter = 0
    numericaldate = datetime.fromisoformat(i).timestamp()
    xData1.append(numericaldate)
    counter += 1

xData = np.array(xData1)

def func(x, a, b, Offset): # Sigmoid A With Offset from zunzun.com
    return  1.0 / (1.0 + numpy.exp(-a * (x-b))) + Offset


# function for genetic algorithm to minimize (sum of squared error)
def sumOfSquaredError(parameterTuple):
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    val = func(xData, *parameterTuple)
    return numpy.sum((yData - val) ** 2.0)


def generate_Initial_Parameters():
    # min and max used for bounds
    maxX = max(xData)
    minX = min(xData)
    maxY = max(yData)
    minY = min(yData)

    parameterBounds = []
    parameterBounds.append([minX, maxX]) # search bounds for a
    parameterBounds.append([minX, maxX]) # search bounds for b
    parameterBounds.append([0.0, maxY]) # search bounds for Offset

    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
    return result.x

# generate initial parameter values
geneticParameters = generate_Initial_Parameters()

# curve fit the test data
fittedParameters, pcov = curve_fit(func, xData, yData, geneticParameters)

print('Parameters', fittedParameters)

modelPredictions = func(xData, *fittedParameters) 

absError = modelPredictions - yData

SE = numpy.square(absError) # squared errors
MSE = numpy.mean(SE) # mean squared errors
RMSE = numpy.sqrt(MSE) # Root Mean Squared Error, RMSE
Rsquared = 1.0 - (numpy.var(absError) / numpy.var(yData))
print('RMSE:', RMSE)
print('R-squared:', Rsquared)



##########################################################
# graphics output section
def ModelAndScatterPlot(graphWidth, graphHeight):
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
    axes = f.add_subplot(111)

    # first the raw data as a scatter plot
    axes.plot(xData, yData,  'D')

    # create data for the fitted equation plot
    xModel = numpy.linspace(min(xData), max(xData))
    yModel = func(xModel, *fittedParameters)

    # now the model as a line plot 
    axes.plot(xModel, yModel)

    axes.set_xlabel('X Data') # X axis data label
    axes.set_ylabel('Y Data') # Y axis data label

    # plt.show()
    # plt.close('all') # clean up after using pyplot

    st.pyplot(plt)

graphWidth = 800
graphHeight = 600
ModelAndScatterPlot(graphWidth, graphHeight)