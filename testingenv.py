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
from matplotlib.pyplot import axis, figure, hist
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
from dtale.views import startup
from streamlit_option_menu import option_menu
from hydralit import HydraHeadApp
from hydralit import HydraApp
import matplotlib.pyplot as plt
import dataingestion

class codetestingenv(HydraHeadApp):

    ico = Functions.set_svg_icon('martechico.ico')
    image=Image.open(ico)
    st.set_page_config(layout="wide",page_title='ABI Analytics', page_icon=image)
        
    title = '<p style="font-family:sans-serif; color:red; font-size: 39px; text-align: center;"><b>Code testing environment</b></p>'
    st.markdown(title, unsafe_allow_html=True)

    ### From Jupyter - 0. Prepare the data

    df, df2, branddf = dataingestion.readdata()
    print(df.head())

        ### End

        ### Enter code to test here

    df, df2, branddf = readdata()

    # components.html('''<style>
    # .zoom {
    #   padding: 50px;
    #   background-color: yellow;
    #   transition: transform .2s; /* Animation */
    #   width: 200px;
    #   height: 200px;
    #   margin: 0 auto;
    # }

    # .zoom:hover {
    #   transform: scale(1.5); /* (150% zoom - Note: if the zoom is too large, it will go outside of the viewport) */
    # }
    # </style>

    # <div class="zoom"></div>''')

    # components.html('''<!DOCTYPE html>
    # <html>
        
    # <head>
    #     <meta charset="UTF-8" />
    #     <meta name="viewport" content=
    #         "width=device-width, initial-scale=1.0" />
            
    #     <title>
    #         How to Zoom an Image on
    #         Mouse Hover using CSS?
    #     </title>
        
    #     <style>
    #         .geeks {
    #             width: 300px;
    #             height: 300px;
    #             overflow: hidden;
    #             margin: 0 auto;
    #         }
        
    #         .geeks img {
    #             width: 100%;
    #             transition: 0.5s all ease-in-out;
    #         }
        
    #         .geeks:hover img {
    #             transform: scale(1.5);
    #         }
    #     </style>
    # </head>
    
    # <body>
    #     <div class="geeks">
    #         <img src=
    # "https://media.geeksforgeeks.org/wp-content/uploads/20200403151026/adblur_gfg.png"
    #             alt="Geeks Image" />
    #     </div>
    # </body>
    
    # </html>''')

    # components.html(
    #     '''
    #     <!DOCTYPE html>
    # <html lang="en">
    # <head>
    # 	<meta charset="UTF-8">
    # 	<meta name="viewport" content="width=device-width, initial-scale=1.0">
    # 	<title>Document</title>
    # 	<style>
    # body{
    # 	padding-top: 50px;
    # }
    # .images{
    # 	display: flex;
    # 	justify-content: space-between;
    # 	align-items: center;
    # }

    # .zoom {
    # transition: transform .2s; /* Animation */
    # width: 150px;
    # height: 150px;
    # margin: 0 auto;
    # background:linear-gradient(rgba(0,0,0,.8),rgba(0,0,0,.4)),  url("Hover.jpeg")  ;
    # background-size: cover;
    # display: flex;
    # justify-content: center;
    # align-items: center;
    # margin-bottom: 50px;
    # border-radius: 50%;
    # }

    # .zoom:hover {
    # transform: scale(1.5); /* (150% zoom - Note: if the zoom is too large, it will go outside of the viewport) */
    # }
    # .zoom:hover + .text .label:after{
    # content:'changed text'; /* (150% zoom - Note: if the zoom is too large, it will go outside of the viewport) */
    # }
    # .zoom:hover > span:after{
    # content:'changed text'; /* (150% zoom - Note: if the zoom is too large, it will go outside of the viewport) */
    # }
    # .zoom > span:after{
    # content: 'hover me';
    # color: #fff;
    # font-size: 1.3em;
    # font-weight: bold;
    # text-transform: capitalize;
    # }
    # /*.label:after{
    # content:'text I want to change';
    # }
    # .label:hover:after{
    # content:'changed text';
    # }*/
    # .newText{
    # text-align: center;
    # margin-top: 50px;
    # }

    # </style>
    # </head>
    # <body>


    # <div class="container">
        
    # 	<div class="images">
            
    # 		<div class="zoom">
    # 		<span></span>
    # 		</div>
    # 		<div class="zoom">
    # 		<span></span>
    # 		</div>
    # 		<div class="zoom">
    # 		<span></span>
    # 		</div>
    # 		<div class="zoom">
    # 		<span></span>
    # 		</div>
    # 		<div class="zoom">
    # 		<span></span>
    # 		</div>
    # 		<div class="zoom">
    # 		<span></span>
    # 		</div>
    # 		<div class="zoom">
    # 		<span></span>
    # 		</div>
    # 		<div class="zoom">
    # 		<span></span>
    # 		</div>
    # 	</div>

    # 	<div class="new-text-container">
            
    # 	</div>

    # </div>


    # <!-- 













    # 	<div class="container1">
    # 	<div class="zoom">
    # 		<span></span>
    # 	</div>

    # <div class="text">
    # 	<p>text <span class="label"></span> text</p>
    # </div>

    # <div class="new-text-container" data-used="no">
    # </div>
    # </div>

    # <div class="container2">
    # 	<div class="zoom">
    # 		<span></span>
    # 	</div>

    # <div class="text">
    # 	<p>text <span class="label"></span> text</p>
    # </div>

    # <div class="new-text-container" data-used="no">
    # </div>
    # </div>

    # <div class="container3">
    # 	<div class="zoom">
    # 		<span></span>
    # 	</div>

    # <div class="text">
    # 	<p>text <span class="label"></span> text</p>
    # </div>

    # <div class="new-text-container" data-used="no">
    # </div>
    # </div>

    # <div class="container4">
    # 	<div class="zoom">
    # 		<span></span>
    # 	</div>

    # <div class="text">
    # 	<p>text <span class="label"></span> text</p>
    # </div>

    # <div class="new-text-container" data-used="no">
    # </div>
    # </div> -->
    # </body>
    # <script>
    # 	// var x=0;
    # 	// document.querySelector(".zoom").addEventListener('click',()=>{ 
    # 	// if(x==0){
    # 	// 	const newDiv = document.createElement("div");
    # 	// newDiv.class="newText";
    #  //  	const newContent = document.createTextNode("Hi there and greetings!");
    #  //  	newDiv.appendChild(newContent);
    #  //  	document.querySelector(".new-text-container").appendChild(newDiv);
    #  //  	// x=1;
    # 	// }
    # 	// });
    # 	const Messages = ["Message1", "Message2", "Message3","Message4","Message5","Message6","Message7","Message8"];
    # 	document.querySelectorAll('.zoom').forEach( (item,index) => {
    #   item.addEventListener('click', () => {
    #   	if(document.querySelector(".newText")){
    #   		document.querySelector(".newText").remove();
    #   	}
    #   	const newDiv = document.createElement("div");
    #   	newDiv.classList.add("newText");;
    #   	const newContent = document.createTextNode(Messages[index]);
    #   	newDiv.appendChild(newContent);
    #   	document.querySelector(`.new-text-container`).appendChild(newDiv);

    #   })
    # })
    # </script>
    # </html>
    #     '''
    # )



    # zoomooz = """<script src="//ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js"></script>
    # <script src="jquery.zoomooz.min.js"></script>
    # <div class="zoomTarget" data-scalemode="width" data-nativeanimation="true"><a href="#more" id='Image 1'><img width='21%' src='https://images.unsplash.com/photo-1565130838609-c3a86655db61?w=200' class='zoom w3-circle w3-hover-opacity'></div>"""

    # components.html(zoomooz)

    # # bootstrap 4 collapse example
    # components.html(
    #     """
    #     <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    #     <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    #     <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    #     <div id="accordion">
    #       <div class="card">
    #         <div class="card-header" id="headingOne">
    #           <h5 class="mb-0">
    #             <button class="btn btn-link" data-toggle="collapse" data-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
    #             Collapsible Group Item #1
    #             </button>
    #           </h5>
    #         </div>
    #         <div id="collapseOne" class="collapse show" aria-labelledby="headingOne" data-parent="#accordion">
    #           <div class="card-body">
    #             Collapsible Group Item #1 content
    #           </div>
    #         </div>
    #       </div>
    #       <div class="card">
    #         <div class="card-header" id="headingTwo">
    #           <h5 class="mb-0">
    #             <button class="btn btn-link collapsed" data-toggle="collapse" data-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
    #             Collapsible Group Item #2
    #             </button>
    #           </h5>
    #         </div>
    #         <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
    #           <div class="card-body">
    #             Collapsible Group Item #2 content
    #           </div>
    #         </div>
    #       </div>
    #     </div>
    #     """,
    #     height=600,
    # )


    # # WORKING ORIGINAL (MINE)

    # content = """
    # <script type="text/javascript">
    #   function reply_click(clicked_id)
    #   {
    #       // alert(clicked_id);
    #   }
    # </script>
    #   <script src="https://cdn.layerjs.org/libs/layerjs/layerjs-0.5.2.min.js" defer=""></script>
    #   <link href="https://cdn.layerjs.org/libs/layerjs/layerjs-0.5.2.css" type="text/css" rel="stylesheet">
    #   <link href="style.css" rel="stylesheet">
    # <style>
    # html, body {
    #   height: 100%;
    # }
    # div {    height: 100px;
    #     text-align: center;
    # }
    # </style>
    # </head>
    # <body height: 1000% lj-type="stage">
    #   <div lj-type="layer" id="content-layer" lj-fit-to="responsive-width" lj-transition="fade" height=1000px>
    #     <div lj-type="frame" id="home" lj-transition="zoomout">
    #       Hello World
    #       <br>
        
    #       <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
    #       <style>
    #     .zoom {
    #     padding: 10px;
    #     width: 210px;
    #     height: 210px;
    #     transition: transform .21s; /* Animation */
    #     margin: 0 auto;
    #     }
    #     .zoom:hover {
    #     transform: scale(1.1); /* (110% zoom - Note: if the zoom is too large, it will go outside of the viewport) */
    #     }
    #     </style>
    #       <a href="#more" id='Image1' onClick="reply_click(this.id)"><img width='21%' src='https://images.unsplash.com/photo-1565130838609-c3a86655db61?w=200' class='zoom w3-circle w3-hover-opacity'>
    #       </a>
    #     </div>
    #     <div lj-type="frame" id="more">
    #       More content...
    #       <br>
    #       <a href="#home&t=2s&p=zoomout">Back to home</a>
        
    #     </div>
    #   </div>
    # """

    # # my_js = f"<script>{content}</script>"

    # components.html(content, height=300)
    # # clicked = click_detector(content)
    # # st.markdown(f"**{clicked} clicked**" if clicked != "" else "**No click**")

    # content2 = """
    # <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
    # // <p><a href='#' id='Link 1'>First link</a></p>
    #    // <p><a href='#' id='Link 2'>Second link</a></p>
    #     <style>
    #     .zoom {
    #     padding: 10px;
    #     width: 210px;
    #     height: 210px;
    #     transition: transform .21s; /* Animation */
    #     margin: 0 auto;
    #     }
    #     .zoom:hover {
    #     transform: scale(1.1); /* (110% zoom - Note: if the zoom is too large, it will go outside of the viewport) */
    #     }
    #     </style>
    #     <div class="w3-container">
    #     <a href='#' id='Image 1'><img width='21%' src='https://images.unsplash.com/photo-1565130838609-c3a86655db61?w=200' class='zoom w3-circle w3-hover-opacity'></a>
    #     // <a href='#' id='Image 2'><img width='21%' src='https://images.unsplash.com/photo-1565372195458-9de0b320ef04?w=200' class='zoom w3-circle w3-hover-opacity'></a>
    #     </div>
    #     """

    # clicked = click_detector(content2)

    # st.markdown(f"**{clicked} clicked**" if clicked != "" else "**No click**")

    # from mycomponent import mycomponent
    # value = mycomponent()
    # st.write("Received", value)

    # # END WORKING ORIGINAL
    
    import streamlit as st
    from click_image_copy_from_demo import st_click_image
    from st_click_detector import click_detector
    import pandas as pd
    from dtale.views import startup
    from streamlit_quill import st_quill

    with st.sidebar:
        choose = option_menu("ABI Analytics", ["Home", "Automate", "Give feedback"],
                            icons=['house', 'joystick', 'keyboard'],
                            menu_icon="app-indicator", default_index=0,
                            styles={
            "container": {"padding": "5!important", "background-color": "#fafafa"},
            "icon": {"color": "black", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#ab0202"},
        }
        )

        if choose == "Automate":
            self.do_redirect('Automated analytics flow')

    value = st_click_image()
    if value is None:
        st.stop()

    st.write("Received", value)

    col1, col2, col3 = st.columns([1,0.36,1])

    clicked = ""

    with col1:
        pass

    with col2:
        if value=='EDA':

            content2 = """
            <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
                <style>
                .zoom {
                padding: 10px;
                width: 210px;
                height: 120px;
                transition: transform .21s; /* Animation */
                margin: 0 auto;
                }
                .zoom:hover {
                transform: scale(1.1); /* (110% zoom - Note: if the zoom is too large, it will go outside of the viewport) */
                }
                </style>
                <div class="w3-container">
                <a href='#' id='Get started'><img width='21%' src='https://i.postimg.cc/SK23bN61/analytics.jpg' class='zoom w3-round-xxlarge w3-hover-opacity'></a>
                <div class="w3-display-middle w3-large"><b><h3>Get started!</h3></b></div>
                </div>
                """

            clicked = click_detector(content2)

            st.markdown(f"**{clicked} clicked**" if clicked != "" else "**No click**")

        else:

            st.write('No picture here')

    with col3:
        pass

    if clicked == "Get started":

        edaenv = st.expander("Guidance on EDA", expanded=False)

        with edaenv:

            st.info("User guide goes here")

            def show_pdf(file_path):
                # Opening tutorial from file path
                with open(file_path, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')

                # Embedding PDF in HTML
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="1500" height="800" type="application/pdf"></iframe>'

                # Displaying File
                st.markdown(pdf_display, unsafe_allow_html=True)

            col1, col2,col3= st.columns(3)
            with col1:  
                if st.button('Read PDF tutorial',key='1'):            
                    show_pdf('.\Automated flow\DtaleInstructions-compressed.pdf')
            with col2:
                st.button('Close PDF tutorial',key='2')                   
            with col3:
                with open(".\Automated flow\DtaleInstructions-compressed.pdf", "rb") as pdf_file:
                    PDFbyte = pdf_file.read()
                st.download_button(label="Download PDF tutorial", key='3',
                        data=PDFbyte,
                        file_name="EDA Instructions.pdf",
                        mime='application/octet-stream')

        startup(data_id="1", data=df2) # All records, no OHE

        if get_instance("1") is None:
            startup(data_id="1", data=df)

        d=get_instance("1")

        # webbrowser.open_new_tab('http://localhost:8501/dtale/main/1') # New window/tab
        # components.html("<iframe src='/dtale/main/1' />", width=1000, height=300, scrolling=True) # Element
        html = f"""<iframe src="/dtale/main/1" height="1000" width="1800"></iframe>""" # Iframe
        # html = "<a href='/dtale/main/1' target='_blank'>Dataframe 1</a>" # New tab link

        st.markdown(html, unsafe_allow_html=True)

        checkbtn = st.button("Validate data")

        if checkbtn == True:
            df_amended = get_instance(data_id="1").data # The amended dataframe ready for upsert
            st.write("Sample of amended data:")
            st.write("")
            st.write(df_amended.head(5))

        clearchanges = st.button("Clear changes made to data")
        if clearchanges == True:
            global_state.cleanup()

        st.write("")
        
        st.subheader("Notes on EDA")

        # Spawn a new Quill editor
        content = st_quill(placeholder="Write your notes here")

        