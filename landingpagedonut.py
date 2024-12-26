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

import streamlit as st

st.set_page_config(layout="wide")

# HTML code for embedding
html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Donut Chart</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center; /* Align the chart closer to the top */
            margin: 0;
            background-color: #f4f4f4;
            position: relative;
            overflow-y: auto; /* Allow scrolling if necessary */
            height: 100vh; /* Ensure the body takes at least the full viewport height */
        }

        canvas {
            width: 80%; /* Stretch the chart to full width */
            height: auto; /* Maintain aspect ratio */
        }

        .info-box {
            position: absolute;
            background: #fff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            font-size: 18px;
            font-weight: bold;
            color: #333;
            width: 240px;
            text-align: center;
            display: none;
            white-space: nowrap; /* Prevent text wrapping */
        }

        .center-image {
            position: absolute;
            width: 290px;
            height: 150px;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            border-radius: 0%;
        }
    </style>
</head>
<body>

<canvas id="donutChart" width="600" height="600"></canvas>
<img src="https://i.postimg.cc/TwhQX0JV/Heuristix-logo.pngg" alt="Cognitive Analytics" class="center-image">
<div class="info-box" id="infoBox">Hover over a segment</div>

<script>
    const ctx = document.getElementById('donutChart').getContext('2d');
    const infoBox = document.getElementById('infoBox');

    const data = {
        labels: [
            '1. Analytical Insights', '2. Domain Expertise', '3. Artificial Intelligence'
        ],
        datasets: [{
            data: [33.33, 33.33, 33.34],
            backgroundColor: [
                'rgba(255, 223, 186, 0.9)', 'rgba(186, 225, 255, 0.9)', 'rgba(186, 255, 240, 0.9)'
            ],
            borderWidth: 1,
            hoverOffset: 20,
        }]
    };

    const descriptions = [
        "This is Analytical Insights, focusing on data analysis.",
        "This is Domain Expertise, emphasizing industry knowledge.",
        "This is Artificial Intelligence, leveraging advanced algorithms."
    ];

    const config = {
        type: 'doughnut',
        data: data,
        options: {
            responsive: false,
            plugins: {
                legend: {
                    display: false, // Hide the legend
                },
                tooltip: {
                    enabled: false, // Disable default tooltips
                }
            },
            onHover: (event, elements) => {
                if (elements.length > 0) {
                    const segmentIndex = elements[0].index;
                    const {x, y} = elements[0].element.tooltipPosition();

                    // Ensure infoBox stays within visible area
                    const bodyRect = document.body.getBoundingClientRect();
                    const infoBoxWidth = infoBox.offsetWidth;
                    const infoBoxHeight = infoBox.offsetHeight;

                    let left = x + 20;
                    let top = y;

                    if (left + infoBoxWidth > bodyRect.width) {
                        left = x - infoBoxWidth - 20;
                    }

                    if (top + infoBoxHeight > bodyRect.height) {
                        top = bodyRect.height - infoBoxHeight - 10;
                    }

                    infoBox.style.left = `${left}px`;
                    infoBox.style.top = `${top}px`;
                    infoBox.style.display = 'block';
                    infoBox.textContent = descriptions[segmentIndex]; // Use the custom description
                } else {
                    infoBox.style.display = 'none';
                }
            },
        },
        plugins: [{
            id: 'doughnutLabels',
            afterDatasetsDraw(chart) {
                const {ctx, data} = chart;
                chart.getDatasetMeta(0).data.forEach((datapoint, index) => {
                    const {x, y} = datapoint.tooltipPosition();
                    ctx.fillStyle = '#000';
                    ctx.font = 'bold 16px Arial'; // Adjusted font size for better fit
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    const lines = data.labels[index].split(' '); // Split label into multiple lines
                    lines.forEach((line, i) => {
                        ctx.fillText(line, x, y - 10 + i * 15); // Adjust line height
                    });
                });
            }
        }]
    };

    new Chart(ctx, config);
</script>

</body>
</html>



"""

# Embed in Streamlit
st.components.v1.html(html_code, height=700, scrolling=False)
