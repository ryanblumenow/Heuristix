from enum import auto
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
from streamlit_extras.switch_page_button import switch_page
import dataingestion
from click_bars import st_click_bars
from app_instance import app  # Import the shared instanc

#add an import to Hydralit
from hydralit import HydraHeadApp
from hydralit import HydraApp

#create a wrapper class
class abianalytics(HydraHeadApp):

#wrap all your code in this method and you should be done

    def run(self):

        ### UNTOUCHED ORIGINAL CODE
        
        # st.session_state['pagechoice'] = 'analytics home'

        with st.sidebar:
            choose = option_menu("ABI Analytics", ["Keep in mind", "Show me the steps", "Give me tips", "Give feedback"],
                                icons=['key', 'bezier2', 'joystick', 'keyboard'],
                                menu_icon="app-indicator", default_index=0,
                                styles={
                "container": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"color": "black", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#ab0202"},
            }
            )

            # print(st.session_state.pagechoice)
            
            if choose == "Keep in mind":
                st.write("Remember to ask good questions. That is the basis of making good decisions.")

            if choose == "Show me the steps" and st.session_state.pagechoice=="test":
                st.write("Steps you should follow:")

            if choose == "Give me tips":
                st.write("Here are some tips:")

            if choose == "Give feedback":
                st.write("Give feedback")
                with st.sidebar.form(key='columns_in_form',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
                    rating=st.slider("Please rate the app", min_value=1, max_value=5, value=3,help='Drag the slider to rate the app. This is a 1-5 rating scale where 5 is the highest rating')
                    text=st.text_input(label='Please leave your feedback here')
                    submitted = st.form_submit_button('Submit')
                    if submitted:
                        st.write('Thanks for your feedback!')
                        st.markdown('Your Rating:')
                        st.markdown(rating)
                        st.markdown('Your Feedback:')
                        st.markdown(text)

        header1 = st.container()
        guidance = st.container()
        dataset = st.container()

        with header1:

            col1,col2,col3 = st.columns([1,3,1])
            with col1:
                pass
            with col2:
                # st.markdown("<h1 style='text-align: center; color: red;'>"'Analytics Playground Home'"</h1>", unsafe_allow_html=True)
                pass
            with col3:
                pass

        with guidance:

            st.markdown("<h3 style='text-align: center; color: green;'>"'Press the button below to understand the analytics flow - and to get started with automating the process.'"</h3>", unsafe_allow_html=True)
            
            col1,col2,col3 = st.columns(3)
            with col1:
                pass
            
            with col2:
                content = """
                <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
                <style>
                .zoom {
                padding: 10px;
                # width: 210px;
                # height: 210px;
                transition: transform .21s; /* Animation */
                margin: 0 auto;
                }
                .zoom:hover {
                transform: scale(1.1); /* (110% zoom - Note: if the zoom is too large, it will go outside of the viewport) */
                }
                </style>
                <div class="w3-container">
                <a href='#' id='Image 1'><img width='100%' src='https://i.postimg.cc/158HHcRn/Automate.png' class='zoom w3-hover-opacity w3-image'></a>
                </div>
                """
                clicked = click_detector(content)

                # st.markdown(f"**{clicked} clicked**" if clicked != "" else "")

            with col3:
                pass
                
            col1,col2,col3 = st.columns([0.5,3,0.5])
            with col1:
                pass
            with col2:
                if clicked == 'Image 1':
                    # st.session_state.pagechoice = 'auto analytics'

                    # self.do_redirect('Automated analytics flow')
                    st.image('.\Automated flow\Analytics flow 2.png')
                    st.info("This is the flow we will follow - you can run any individual model from the menu above or click on the 'Automated analytics' tab to start the guided process.")
                    
                    # switch_page('hlautoanalytics')

        with dataset:

            # df, df2, branddf = dataingestion.readdata()
            df = dataingestion.readdata()
            print(df.head())

            col1, col2, col3 = st.columns([1,3,1])
            
            with col1:
                pass
            with col2:
                # st.image('.\Automated flow\Analytics flow 2.png')
                pass
            with col3:
                pass

        # def run(self):
        clicked_bar = st_click_bars()  # Get the value from the custom component
        st.write(f"Clicked Bar: {clicked_bar}")  # Optional: Debugging
        st.write("Navbar State Before Redirect:", st.session_state.get('selected_app'))
        
        # Redirect based on the clicked bar
        if clicked_bar == "Bar1":
            st.session_state['selected_app'] = "Heuristix home"
            self.do_redirect("Heuristix home")  # Replace "Dwell" with your target page/tab name
            st.experimental_rerun()
            # app.session_state['mainHydralitMenuComplex'] = "Heuristix home"
        elif clicked_bar == "Bar2":
            self.do_redirect("ABI Analytics home")  # Replace "Dwell 2" with your target page/tab name
        elif clicked_bar == "Bar3":
            self.do_redirect("Automated analytics flow")  # Replace with your target page/tab name


        # value2 = st_click_bars()
        # if value2 is None:
        #     # st.stop()
        #     pass

        # st.success("{} selected".format(value2))

    ### UNTOUCHED ORIGINAL CODE