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
from bioinfokit.analys import stat
import statsmodels.api as sm
from statsmodels.graphics.factorplots import interaction_plot
from hlautoanalytics import autoanalytics
from st_click_detector import click_detector
from streamlit_option_menu import option_menu
from streamlit_quill import st_quill
import dataingestion

#add an import to Hydralit
from hydralit import HydraHeadApp
from hydralit import HydraApp

st.subheader("Hypothesis testing on characteristics with optional disaggregation by brand or brand groups")

df, df2, branddf = dataingestion.readdata()

with st.spinner('Please wait while we conduct the ensemble methods analysis using the random forest algorithm'):

    my_bar = st.progress(0)

    time.sleep(10)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)

    start_time = time.time()

    st.info("This ensemble methods random forest model examines the effect of gender, age, and city of consumer on brand preference. It is a work in progress and will be expanded soon.")

    # test classification dataset

    st.write("Test classification dataset:")

    # define dataset
    X = df.loc[0:10000,["GENDER", "AGE", "CITY"]] # Subsampling for speed and efficiency, dynamic to certain relevant explanatory variables (could combine with dimensionality reduction or with conjoint analysis above)
    y = df2.BRAND.head(10001) #.apply(lambda x: '1' if '*astle' in x else 0).head(10001) # Dynamic to brand, subsampling for speed and efficiency
    # summarize the dataset
    st.write("The shape of your subsampled data, in the form of explanatory variables (gender, age, city), and predicted variable (brand):")
    st.write(X.shape, y.shape)

    # evaluate random forest algorithm for classification

    st.write("")

    st.write("Evaluate random forest algorithm for classification:")

    # define the model
    model = RandomForestClassifier()
    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    st.write('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    # random forest for classification
    # define 

    st.write("")
    
    st.write("Model outputs:")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # define the model
    model = RandomForestClassifier()
    # fit the model on the training dataset
    model.fit(X_train, y_train)
    # make a single prediction
    print(X)
    row = X.head(1)
    yhat = model.predict(row)
    st.write('Predicted Class: ' + yhat[0])

    st.write("")

    # test regression dataset
    st.write("Test regression dataset:")
    # define dataset
    # Choose predicted variable - this will become dynamic in the app
    y = df['BRAND'].head(10000)
    # Define predictor variables
    X = df.iloc[:, 1:-2].head(10000)
    # X, y = np.array(x), np.array(y)
    # summarize the dataset
    st.write("The shape of your subsampled data, in the form of explanatory variables (gender, age, city), and predicted variable (brand):")
    st.write(X.shape, y.shape)
    
    print("X df")
    print(X.head(5))

    st.write("")

    # evaluate random forest ensemble for regression
    st.write("Evaluate random forest ensemble for regression:")
    # define the model
    model = RandomForestRegressor()
    # evaluate the model
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    st.write('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    st.write("")

    # random forest for making predictions for regression
    st.write("Making predictions for regression:")
    # define dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # define the model
    model = RandomForestRegressor()
    # fit the model on the whole dataset
    model.fit(X_train, y_train)
    # make a single prediction
    row = X.head(1)
    yhat = model.predict(row)
    st.write('Prediction: %d' % yhat[0])

    st.write("")

    st.write("")

    # Spawn a new Quill editor
    st.subheader("Notes on random forest analysis")
    randomforestcontent = st_quill(placeholder="Write your notes here")

    st.write("Conducting the random forest model took ", time.time() - start_time, "seconds to run")

    st.write("")

tunerfparams = st.button("Click here to tune the hyperparameters for the random forest model")

if tunerfparams == True:

    with st.spinner('Please wait while we tune the hyperparameters for the random forest model'):

        my_bar = st.progress(0)

        time.sleep(10)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)

        start_time = time.time()

        # Tuning hyperparameters for random forest

        # 1

        # Bootstrap sizes of 10-100% on random forest algorithm
        # Explore random forest bootstrap sample size on performance
        # Bootstrap sample size that is equal to the size of the training dataset generally achieves the best results and is the default

        st.write("Explore bootstrap sizes of '10-100%' on random forest algorithm")
        st.write("Bootstrap sample size that is equal to the size of the training dataset generally achieves the best results and is the default")

        # get the dataset
        def get_dataset():
            X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
            return X, y

        # get a list of models to evaluate
        def get_models():
            models = dict()
            # explore ratios from 10% to 100% in 10% increments
            for i in np.arange(0.1, 1.1, 0.1):
                key = '%.1f' % i
                # set max_samples=None to use 100%
                if i == 1.0:
                    i = None
                models[key] = RandomForestClassifier(max_samples=i)
            return models

        # evaluate a given model using cross-validation
        def evaluate_model(model, X, y):
            # define the evaluation procedure
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            # evaluate the model and collect the results
            scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
            return scores

        # define dataset
        X, y = get_dataset()
        # get the models to evaluate
        models = get_models()
        # evaluate the models and store results
        results, names = list(), list()
        for name, model in models.items():
            # evaluate the model
            scores = evaluate_model(model, X, y)
            # store the results
            results.append(scores)
            names.append(name)
            # summarize the performance along the way
            st.write('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
        # plot model performance for comparison
        pyplot.boxplot(results, labels=names, showmeans=True)
        pyplot.show()

        buf1 = BytesIO()
        plt.savefig(buf1, format="png")
        st.image(buf1)

        # 2

        # Explore random forest number of features effect on performance

        st.write("")

        st.write("Explore random forest number of features effect on performance")

        # get the dataset
        def get_dataset():
            X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
            return X, y

        # get a list of models to evaluate
        def get_models():
            models = dict()
            # explore number of features from 1 to 21
            for i in range(1,22):
                models[str(i)] = RandomForestClassifier(max_features=i)
            return models

        # evaluate a given model using cross-validation
        def evaluate_model(model, X, y):
            # define the evaluation procedure
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            # evaluate the model and collect the results
            scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
            return scores

        # define dataset
        X, y = get_dataset()
        # get the models to evaluate
        models = get_models()
        # evaluate the models and store results
        results, names = list(), list()
        for name, model in models.items():
            # evaluate the model
            scores = evaluate_model(model, X, y)
            # store the results
            results.append(scores)
            names.append(name)
            # summarize the performance along the way
            st.write('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
        # plot model performance for comparison
        pyplot.boxplot(results, labels=names, showmeans=True)
        pyplot.show()

        buf2 = BytesIO()
        plt.savefig(buf2, format="png")
        st.image(buf2)

        # 3

        # Explore random forest number of trees effect on performance

        st.write("")

        st.write("Explore random forest number of trees effect on performance")

        # get the dataset
        def get_dataset():
            X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
            return X, y

        # get a list of models to evaluate
        def get_models():
            models = dict()
            # define number of trees to consider
            n_trees = [10, 50, 100, 500, 1000]
            for n in n_trees:
                models[str(n)] = RandomForestClassifier(n_estimators=n)
            return models

        # evaluate a given model using cross-validation
        def evaluate_model(model, X, y):
            # define the evaluation procedure
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            # evaluate the model and collect the results
            scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
            return scores

        # define dataset
        X, y = get_dataset()
        # get the models to evaluate
        models = get_models()
        # evaluate the models and store results
        results, names = list(), list()
        for name, model in models.items():
            # evaluate the model
            scores = evaluate_model(model, X, y)
            # store the results
            results.append(scores)
            names.append(name)
            # summarize the performance along the way
            st.write('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
        # plot model performance for comparison
        pyplot.boxplot(results, labels=names, showmeans=True)
        pyplot.show()

        buf3 = BytesIO()
        plt.savefig(buf3, format="png")
        st.image(buf3)

        # 4

        # Explore random forest tree depth effect on performance

        st.write("")

        st.write("Explore random forest tree depth effect on performance")

        # get the dataset
        def get_dataset():
            X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
            return X, y

        # get a list of models to evaluate
        def get_models():
            models = dict()
            # consider tree depths from 1 to 7 and None=full
            depths = [i for i in range(1,8)] + [None]
            for n in depths:
                models[str(n)] = RandomForestClassifier(max_depth=n)
            return models

        # evaluate a given model using cross-validation
        def evaluate_model(model, X, y):
            # define the evaluation procedure
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            # evaluate the model and collect the results
            scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
            return scores

        # define dataset
        X, y = get_dataset()
        # get the models to evaluate
        models = get_models()
        # evaluate the models and store results
        results, names = list(), list()
        for name, model in models.items():
            # evaluate the model
            scores = evaluate_model(model, X, y)
            # store the results
            results.append(scores)
            names.append(name)
            # summarize the performance along the way
            st.write('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
        # plot model performance for comparison
        pyplot.boxplot(results, labels=names, showmeans=True)
        pyplot.show()

        buf4 = BytesIO()
        plt.savefig(buf4, format="png")
        st.image(buf4)

        st.write("")

        st.write("Tuning hyperparameters for the random forest model took ", time.time() - start_time, "seconds to run")
