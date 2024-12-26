from os import curdir
import hydralit as hy
from numpy.core.fromnumeric import var
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
from numpy import mean
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


title = '<p style="font-family:sans-serif; color:red; font-size: 39px; text-align: center;"><b>Custom data/code testing environment</b></p>'
st.markdown(title, unsafe_allow_html=True)

x=0
y=0

x = st.selectbox("Please choose first variable:", vars, key="correlx", index=x)
y = st.selectbox("Please choose second variable:", vars, key="correly", index=y)

# var1 = df[x].iloc[1:101]
# var2 = df[y].iloc[1:101]

var1 = df[x].sample(10000)
var2 = df[y].sample(10000)

st.subheader("Correlation between chosen 2 variables:" + " " + x + " and " + y)

st.write("Correlation coefficient: ", var1.corr(var2))
st.write("Pearson correlation coefficient: ", pearsonr(var1, var2))
st.write("Spearman correlation coefficient: ", spearmanr(var1, var2))

### From Jupyter: 1. Hypothesis testing: correlation

st.subheader("Correlation matrix, Pearson")

st.write(df.corr(method = 'pearson'))

with st.expander("Interpretation guide"):
    st.write("Variables within a dataset can be related for many reasons.\n\n"
    "For example:\n"
    "One variable could cause or depend on the values of another variable.\n"
    "One variable could be lightly associated with another variable.\n"
    "Two variables could depend on a third unknown variable.\n\n"
    "It can be useful in data analysis and modeling to better understand the relationships between variables. The statistical relationship between two variables is referred to as their correlation.\n"
    "A correlation could be positive, meaning both variables move in the same direction, or negative, meaning that when one variable's value increases, the other variables' values decrease. Correlation can also be neutral or zero, meaning that the variables are unrelated.")
    st.info("Positive Correlation: both variables change in the same direction.\n\n"
    "Neutral Correlation: No relationship in the change of the variables.\n\n"
    "Negative Correlation: variables change in opposite directions.\n\n"
    "A correlation coefficient closer to 1 or to -1 indicates stronger relationships between the variables, while a correlation coefficient closer to 0 indicates a lesser relationship between the variables.")
    st.write("The performance of some algorithms can deteriorate if two or more variables are tightly related, called multicollinearity. An example is linear regression, where one of the offending correlated variables should be removed in order to improve the skill of the model.\n\n"
    "We may also be interested in the correlation between input variables with the output variable in order provide insight into which variables may or may not be relevant as input for developing a model.\n"
    "The structure of the relationship may be known, e.g. it may be linear, or we may have no idea whether a relationship exists between two variables or what structure it may take. Depending what is known about the relationship and the distribution of the variables, different correlation scores can be calculated.")

### End

st.write("")

st.subheader("Notes on correlation analysis")

# Spawn a new Quill editor
content = st_quill(placeholder="Write your notes here", key="correlquill")