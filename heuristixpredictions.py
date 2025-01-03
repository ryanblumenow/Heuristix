import streamlit as st
from streamlit_option_menu import option_menu
import base64
from email import header
from html.entities import html5
from importlib.resources import read_binary
# import hydralit as hy
from markdown import markdown
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
# import tkinter
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
from matplotlib.pyplot import axis, hist
from scipy import stats as stats
# from bioinfokit.analys import stat
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
from statsmodels.graphics.factorplots import interaction_plot
from sklearn.decomposition import PCA
from st_click_detector import click_detector
from streamlit.components.v1 import html
import streamlit.components.v1 as components
import pandas as pd
import streamlit as st
from click_image_copy_from_demo import st_click_image
from dtale.views import startup
from streamlit_quill import st_quill
# import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import dataingestion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(layout="wide")

def predict():

    st.title("Make a prediction")
    
    st.subheader("Logistic regression")

    with st.spinner('Please wait while we conduct the logistic regression analysis'):

        my_bar = st.progress(0)

        time.sleep(10)

        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)

        start_time = time.time()

        st.write("**This model evaluates the probability of different classes for a chosen categorical dependent variable.**")

        df = dataingestion.readdata()

        with st.expander("Sample of data", expanded=False):
            samplecol1, samplecol2, samplecol3 = st.columns([1,3,1])
            with samplecol2:
                st.markdown("### <center>Sample of data:</center>", unsafe_allow_html=True)
                st.write(df.sample(8))
        add_vertical_space(3)

        st.info("Please select a variable for predictions - this should be categorical, as we will attempt to determine which category in this variable is most likely for a hypothetical individual you create.")

        ### Select dependent variable
        dependent_var = st.selectbox(
            "Please select the dependent categorical variable for predictions",
            df.select_dtypes(include=['object', 'category']).columns
        )

        st.write(f"Selected dependent variable: **{dependent_var}**")

        # Logistic regression assumptions
        with st.expander("Logistic regression assumptions"):
            st.info("Basic assumptions must be met for logistic regression: independence of errors, linearity in the logit for continuous variables, absence of multicollinearity, and lack of strongly influential outliers.")
            st.info("Differences with linear regression include that logistic regression does not require a linear relationship between dependent and independent variables, residuals do not need to be normally distributed, and homoscedasticity is not required. The dependent variable must be categorical.")
            link = f'["See more"]("https://www.statology.org/assumptions-of-logistic-regression/")'
            st.markdown(link, unsafe_allow_html=True)

        st.info("We can interpret the coefficients of the logistic regression as the probability of an observation falling into a particular class of dependent variable given changes in a set of independent, or input, variables. This means we can predict the potential outcome class for a relevant variable based on how other variables in the dataset change.")
        st.write("")
        st.info("Please set up a hypothetical consumer to determine their likely classification, with characteristics derived from the dataset:")

        # # Collect hypothetical consumer inputs
        # age = st.number_input("Please enter consumer age")
        # gender = st.radio("Please select consumer gender", options=["Male", "Female"])
        # firstint = st.number_input("How many days ago was the consumer's first interaction?")
        # lastint = st.number_input("How many days ago was the consumer's last interaction?")
        # province = st.radio("Please select consumer province", ["Eastern Cape", "Free State", "Gauteng", "KwaZulu-Natal", "Limpopo", "Mpumalanga", "Northern Cape", "North West", "Western Cape"])
        # email = st.radio("Has the consumer opted in for email notifications?", options=["Yes", "No"])
        # sms = st.radio("Has the consumer opted in for SMS notifications?", options=["Yes", "No"])
        # push = st.radio("Has the consumer opted in for push notifications?", options=["Yes", "No"])

        # gender = 1 if gender == "Male" else 2
        # province = ["Eastern Cape", "Free State", "Gauteng", "KwaZulu-Natal", "Limpopo", "Mpumalanga", "Northern Cape", "North West", "Western Cape"].index(province) + 1
        # email, sms, push = (1 if x == "Yes" else 2 for x in [email, sms, push])

        # Dictionary to store user inputs for the independent variables
        hypothetical_individual = {}

        # Dictionary to store label encoders for categorical variables
        label_encoders = {}

        # Create input fields for all variables except the dependent variable
        for column in df.columns:
            if column == dependent_var:
                continue  # Skip the dependent variable

            col_dtype = df[column].dtype

            if pd.api.types.is_numeric_dtype(col_dtype):
                # Numeric input with range and default value
                min_val = float(df[column].min())
                max_val = float(df[column].max())
                default_val = float(df[column].mean())
                hypothetical_individual[column] = st.number_input(
                    f"{column} (numeric, range: {min_val} - {max_val})",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                )
            elif pd.api.types.is_categorical_dtype(col_dtype) or pd.api.types.is_object_dtype(col_dtype):
                # Categorical input with options from unique values
                unique_values = df[column].dropna().unique()
                hypothetical_individual[column] = st.selectbox(
                    f"{column} (categorical)",
                    options=unique_values,
                )

                # Prepare label encoder for backend encoding
                le = LabelEncoder()
                le.fit(df[column].dropna())  # Fit on the column's values
                label_encoders[column] = le
            else:
                st.warning(f"Unsupported data type for column {column}. Skipped.")

        # Display the user's hypothetical individual and the constructed row
        st.write("Hypothetical individual setup:")
        st.json(hypothetical_individual)

        # Ensure categorical variables in `row` are encoded for model prediction
        encoded_row = []
        for column in df.columns:
            if column == dependent_var:
                continue  # Skip the dependent variable
            value = hypothetical_individual[column]
            if column in label_encoders:  # Encode categorical variables
                value = label_encoders[column].transform([value])[0]
            encoded_row.append(value)

        # Encode the dependent variable if it's categorical or object dtype
        le_dependent = None
        if pd.api.types.is_categorical_dtype(df[dependent_var]) or pd.api.types.is_object_dtype(df[dependent_var]):
            le_dependent = LabelEncoder()
            le_dependent.fit(df[dependent_var].dropna())  # Fit on the dependent variable

        # Define dataset
        dflogreg = df
        ylogreg = dflogreg[dependent_var].sample(10000)
        Xlogreg = dflogreg.drop(columns=[dependent_var]).sample(10000)

        # Encode data if necessary
        Xlogreg = pd.get_dummies(Xlogreg)
        if le_dependent:
            ylogreg = le_dependent.transform(ylogreg)  # Transform ylogreg with the encoder
        else:
            ylogreg = pd.factorize(ylogreg)[0]

        # Define and evaluate the logistic regression model
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=1)
        n_scores = cross_val_score(model, Xlogreg, ylogreg, scoring='accuracy', cv=cv, n_jobs=-1)

        st.write('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

        # Train and predict
        X_train, X_test, y_train, y_test = train_test_split(Xlogreg, ylogreg, test_size=0.25, random_state=42)
        model.fit(X_train, y_train)

        # Apply the same transformations to the hypothetical individual
        input_df = pd.DataFrame([hypothetical_individual])  # Convert the hypothetical individual into a DataFrame
        input_df_encoded = pd.get_dummies(input_df)  # Apply one-hot encoding

        # Align the encoded row with the training features (Xlogreg.columns)
        missing_cols = set(Xlogreg.columns) - set(input_df_encoded.columns)
        for col in missing_cols:
            input_df_encoded[col] = 0  # Add missing columns with default value 0
        input_df_encoded = input_df_encoded[Xlogreg.columns]  # Reorder columns to match training data

        # Convert to a row format
        encoded_row = input_df_encoded.iloc[0].values

        # Make the prediction
        yhat_encoded = model.predict([encoded_row])[0]

        # Decode the predicted class if a LabelEncoder was used
        if le_dependent:
            yhat = le_dependent.inverse_transform([yhat_encoded])[0]
        else:
            yhat = yhat_encoded  # If no encoder, use raw prediction

        add_vertical_space(3)

        # Display the decoded predicted class
        st.markdown(f"### <span style='background-color: #ffcc66; color: black; padding: 5px; border-radius: 5px;'>Our prediction is that this individual will likely exhibit characteristics similar to that of {yhat} for {dependent_var}.</span>", unsafe_allow_html=True)

        add_vertical_space(3)
        
        # Predicted probabilities
        st.write("**Predict a multinomial probability distribution for the dependent variable, based on the whole dataset:**")
        yhat_proba = model.predict_proba([encoded_row])
        st.write('Predicted probabilities:')
        st.write(yhat_proba[0].tolist())

        # Probability plot
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(yhat_proba[0].tolist())
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)

        # Notes section
        st.subheader("Notes on logistic regression analysis")
        logregcontent = st_quill(placeholder="Write your notes here", key="logregquill")

        st.write("Logistic regression took ", time.time() - start_time, "seconds to run")

    # Hyperparameter tuning button
    tunelogreg = st.button("Tune penalty hyperparameter")

    if tunelogreg:
        with st.spinner('Please wait while we tune the hyperparameters for the logistic regression analysis'):

            my_bar = st.progress(0)

            time.sleep(10)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)

            start_time = time.time()

            # Define models for tuning
            def get_models():
                models = dict()
                for p in [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]:
                    key = '%.4f' % p
                    if p == 0.0:
                        models[key] = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='none')
                    else:
                        models[key] = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=p)
                return models

            # Evaluate a model
            def evaluate_model(model, Xlogreg, ylogreg):
                cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
                scores = cross_val_score(model, Xlogreg, ylogreg, scoring='accuracy', cv=cv, n_jobs=-1)
                return scores

            models = get_models()
            results, names = list(), list()
            for name, model in models.items():
                scores = evaluate_model(model, Xlogreg, ylogreg)
                results.append(scores)
                names.append(name)
                st.write('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

            # Plot results
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.boxplot(results, labels=names, showmeans=True)
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)

            st.write("Smaller C = larger penalty; our results show we need a larger penalty for better model performance.")

            st.write("Tuning hyperparameters for the logistic regression took ", time.time() - start_time, "seconds to run")
