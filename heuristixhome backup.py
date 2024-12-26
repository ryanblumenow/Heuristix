import streamlit as st
from streamlit_option_menu import option_menu
import base64
from email import header
from html.entities import html5
from importlib.resources import read_binary
import hydralit as hy
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
from matplotlib.pyplot import axis, hist
from scipy import stats as stats
from bioinfokit.analys import stat
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
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import dataingestion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
import matplotlib.colors as mcolors

#add an import to Hydralit
from hydralit import HydraHeadApp

#create a wrapper class
class heuristixhome(HydraHeadApp):

#wrap all your code in this method and you should be done
    def run(self):
        #-------------------existing untouched code------------------------------------------

        # title = '<p style="font-family:sans-serif; color:red; font-size: 39px; text-align: center;"><b>Automated analytics flow</b></p>'
        # st.markdown(title, unsafe_allow_html=True)

        # HTML code for embedding
        html_code = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Interactive Bars with Images</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    display: flex;
                    flex-direction: column; /* Stack chart and logo vertically */
                    justify-content: center;
                    align-items: center;
                    margin: 0;
                    height: 100vh;
                    background-color: #f4f4f4;
                }

                .chart-container {
                    display: flex;
                    align-items: flex-end;
                    gap: 20px;
                    position: relative;
                }

                .bar {
                    width: 100px;
                    position: relative;
                    transition: transform 0.3s ease;
                    cursor: pointer;
                }

                .bar img {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                    border-radius: 10px;
                }

                .bar:hover {
                    transform: scale(1.1);
                }

                .tooltip {
                    position: absolute;
                    background: #fff;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
                    font-size: 18px;
                    font-weight: bold;
                    color: #333;
                    width: 300px;
                    text-align: center;
                    display: none;
                    pointer-events: none;
                    white-space: normal;
                    z-index: 10;
                }

                .logo-container {
                    margin-top: 20px; /* Add space between bars and logo */
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }

                .logo-container img {
                    width: 390px; /* Adjust size of the logo */
                    height: auto;
                    object-fit: contain;
                }


            </style>
        </head>
        <body>

        <div class="chart-container">
            <div class="bar" style="height: 200px;" data-description="This is Analytical Insights, focusing on data analysis.">
                <img src="https://i.postimg.cc/67YnrDNB/edalbl.jpg" alt="Analytical Insights">
            </div>
            <div class="bar" style="height: 300px;" data-description="This is Domain Expertise, emphasizing industry knowledge.">
                <img src="https://i.postimg.cc/68xrtr0x/correllbl.jpg" alt="Domain Expertise">
            </div>
            <div class="bar" style="height: 400px;" data-description="This is Artificial Intelligence, leveraging advanced algorithms.">
                <img src="https://i.postimg.cc/94TszY3P/hyptestinglbl.jpg" alt="Artificial Intelligence">
            </div>
            <div class="tooltip" id="tooltip"></div>
        </div>

        <div class="logo-container">
            <img src="https://i.postimg.cc/TwhQX0JV/Heuristix-logo.png" alt="Heuristix">
        </div>


        <script>
            const bars = document.querySelectorAll('.bar');
            const tooltip = document.getElementById('tooltip');

            bars.forEach(bar => {
                bar.addEventListener('mouseenter', (e) => {
                    const description = bar.getAttribute('data-description');
                    tooltip.textContent = description;
                    tooltip.style.display = 'block';

                    // Position tooltip directly over the bar
                    const rect = bar.getBoundingClientRect();
                    tooltip.style.left = `${rect.left + rect.width / 2 - tooltip.offsetWidth / 2 - 500}px`;
                    tooltip.style.top = `${rect.top - tooltip.offsetHeight - 10}px`; // Slightly above the bar
                });

                bar.addEventListener('mouseleave', () => {
                    tooltip.style.display = 'none';
                });
            });
        </script>

        </body>
        </html>

        """

        # Embed in Streamlit
        st.components.v1.html(html_code, height=700, scrolling=False)

        st.write("Navbar State After Redirect:", st.session_state.get('selected_app'))
        
        df = dataingestion.readdata()
