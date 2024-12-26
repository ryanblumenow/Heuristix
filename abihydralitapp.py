from hydralit import HydraApp
import streamlit as st
from hlabianalyticshome import abianalytics
from hlabitraining import training
from hlcheatsheet import cheatsheet
from hlforecasting import forecasting
from hlcustomdata import customdataenv
from hlautoanalytics import autoanalytics
from hlindivmodel import indivmodels
from heuristixhome import heuristixhome
# from streamlit.state.session_state import SessionState
from PIL import Image
from functions import *
import keyboard
import os
import signal
import streamlit.components.v1 as components
import hydralit_components as hl
import hydralit_components as hc
import time
from hlautoanalytics import autoanalytics
import apps
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
from app_instance import app  # Import the shared instanc

if __name__ == '__main__':

    # if 'pagechoice' not in st.session_state:
    #     st.session_state['pagechoice'] = 'analytics'

    #this is the host application, we add children to it and that's it!
    app = HydraApp(title='ABI Analytics', favicon="dartico.jpg", hide_streamlit_markers=False, layout='wide', navbar_sticky=True) #, navbar_sticky=True, navbar_mode='sticky', use_navbar=True)

    # To eliminate space at top of page

    hide_streamlit_style = """
    <style>
        #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 3rem;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Customized footer

    customizedfooter = """
            <style>
            footer {
	
	visibility: hidden;
	
	}
    footer:after {
        content:'Made by and (c) Ryan Blumenow';
        visibility: visible;
        display: block;
        position: relative;
        #background-color: red;
        padding: 5px;
        top: 2px;
        left: 630px;
    }</style>"""

    st.markdown(customizedfooter, unsafe_allow_html=True)

    # Right-hand-side sidebar

    # html = """
    #             <style>
    #                 .reportview-container {
    #                 flex-direction: row-reverse;
    #                 }

    #                 header > .toolbar {
    #                 flex-direction: row-reverse;
    #                 left: 1rem;
    #                 right: auto;
    #                 }

    #                 .sidebar .sidebar-collapse-control,
    #                 .sidebar.--collapsed .sidebar-collapse-control {
    #                 left: auto;
    #                 right: 0.5rem;
    #                 }

    #                 .sidebar .sidebar-content {
    #                 transition: margin-right .3s, box-shadow .3s;
    #                 }

    #                 .sidebar.--collapsed .sidebar-content {
    #                 margin-left: auto;
    #                 margin-right: -21rem;
    #                 }

    #                 @media (max-width: 991.98px) {
    #                 .sidebar .sidebar-content {
    #                     margin-left: auto;
    #                 }
    #                 }
    #             </style>
    #             """
    # st.markdown(html, unsafe_allow_html=True) # This line and the html above are for the right-hand sidebar

    # side_bar = """
    # <style>
    #     /* The whole sidebar */
    #     .css-1lcbmhc.e1fqkh3o0{
    #     margin-top: 3.8rem;
    #     }
        
    #     /* The display arrow */
    #     .css-sg054d.e1fqkh3o3 {
    #     margin-top: 5rem;
    #     }

    #     /* The display arrow */
    #     .css-bauj2f.e1fqkh3o3 {
    #     margin-top: 5rem;
    #     }

    # </style> 
    # """
    # st.markdown(side_bar, unsafe_allow_html=True) # This moves the sidebar down to accommodate the navigation bar at the top

    # st.title("New Sidebar")
    # st.sidebar.text("Hyperparameter tuning and")
    # st.sidebar.text("model optimization")

    from st_on_hover_tabs import on_hover_tabs

    st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)

    with st.sidebar:
            st.image('ttest.png')
            add_vertical_space(1)
            value = on_hover_tabs(tabName=['KPI Overview', 'Dashboard/Analytics'], 
                                iconName=['contacts', 'dashboard', 'account_tree', 'table', 'report', 'edit', 'update', 'pivot_table_chart', 'menu_book'],
                                styles = {'navtab': {'background-color':'#6d0606',
                                                    'color': 'white',
                                                    'font-size': '18px',
                                                    'transition': '.3s',
                                                    'white-space': 'nowrap',
                                                    'text-transform': 'uppercase'},
                                        'tabOptionsStyle': {':hover :hover': {'color': 'red',
                                                                        'cursor': 'pointer'}},
                                        'iconStyle':{'position':'fixed',
                                                        'left':'7.5px',
                                                        'text-align': 'left'},
                                        'tabStyle' : {'list-style-type': 'none',
                                                        'margin-bottom': '30px',
                                                        'padding-left': '30px'}},
                                key="hoversidebar",
                                default_choice=0)

    css = '''
    <style>
        .stTabs [data-baseweb="tab-highlight"] {
            background-color:blue;
        }
    </style>
    '''

    st.markdown(css, unsafe_allow_html=True)

    preheader = st.container()
    header1 = st.container()

    # with preheader:
    #     components.html('''
    #     <!DOCTYPE html>
    #     <html>

    #     <head>
    #         <meta name="viewport" content="width=device-width, initial-scale=1">
    #         <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons">

    #         <style>
    #             body {
    #                 font-family: "Lato", sans-serif;
    #             }
                
    #             .sidebar {
    #                 height: 100%;
    #                 width: 39px;
    #                 position: fixed;
    #                 z-index: 1;
    #                 top: 0;
    #                 left: 0;
    #                 background-color: #A7A9AA;
    #                 overflow-x: hidden;
    #                 transition: 0.5s;
    #                 padding-top: 60px;
    #                 white-space: nowrap;
    #             }
                
    #             .sidebar a {
    #                 padding: 1px 1px 1px 8px;
    #                 text-decoration: none;
    #                 font-size: 18px;
    #                 color: #818181;
    #                 display: block;
    #                 transition: 0.3s;
    #             }
                
    #             .sidebar a:hover {
    #                 color: #f1f1f1;
    #             }
                
    #             .sidebar .closebtn {
    #                 position: absolute;
    #                 top: 0;
    #                 right: 14px;
    #                 font-size: 14px;
    #                 margin-left: 5px;
    #             }
                
    #             .material-icons,
    #             .icon-text {
    #                 vertical-align: middle;
    #             }
                
    #             .material-icons {
    #                 padding-bottom: 1px;
    #             }
                
    #             #main {
    #                 transition: margin-left .5s;
    #                 padding: 19px;
    #                 margin-left: 39px;
    #             }
    #             /* On smaller screens, where height is less than 450px, change the style of the sidenav (less padding and a smaller font size) */
                
    #             @media screen and (max-height: 450px) {
    #                 .sidebar {
    #                     padding-top: 5px;
    #                 }
    #                 .sidebar a {
    #                     font-size: 14px;
    #                 }
    #             }
    #         </style>
    #     </head>

    #     <body>

    #         <div id="mySidebar" class="sidebar" onmouseover="toggleSidebar()" onmouseout="toggleSidebar()">
    #             <a href="#"><span><i class="material-icons">info</i><span class="icon-text">&nbsp;&nbsp;&nbsp;&nbsp;About</span></a><br>
    #             <a href="#"><i class="material-icons">spa</i><span class="icon-text"></span>&nbsp;&nbsp;&nbsp;&nbsp;Services</a></span>
    #             </a><br>
    #             <a href="#"><i class="material-icons">email</i><span class="icon-text"></span>&nbsp;&nbsp;&nbsp;&nbsp;Contact<span></a>
    #         </div>

    #         <div id="main">
    #             <h2></h2>
    #             <p></p>
    #             <p></p>
    #         </div>

    #         <script>
    #             var mini = true;

    #             function toggleSidebar() {
    #                 if (mini) {
    #                     console.log("opening sidebar");
    #                     document.getElementById("mySidebar").style.width = "250px";
    #                     document.getElementById("main").style.marginLeft = "250px";
    #                     this.mini = false;
    #                 } else {
    #                     console.log("closing sidebar");
    #                     document.getElementById("mySidebar").style.width = "39px";
    #                     document.getElementById("main").style.marginLeft = "39px";
    #                     this.mini = true;
    #                 }
    #             }
    #         </script>

    #     </body>

    #     </html>''')

    # with header1:
    #     clm1, clm2, clm3, clm4, clm5 = st.columns(5)
    #     with clm1:
    #         pass
    #     with clm2:
    #         pass
    #     with clm3:
    #         image1 = Functions.set_svg_image('Martechlogo.jpg')
    #         image2 = Image.open(image1)
    #         st.image(image2, width=283)
    #     with clm4:
    #         pass
    #     with clm5:
    #         quitapp = st.button("Exit", key="multipgexit")
    
    # if quitapp==True:
    #     keyboard.press_and_release('ctrl+w')
    #     os.kill(os.getpid(), signal.SIGTERM)

    #add all your application classes here
    app.add_app(title="Heuristix home",icon="analytics.ico", app=heuristixhome())
    app.add_app(title="ABI Analytics home",icon="analytics.ico", app=abianalytics())
    app.add_app(title="Training", icon="training.webp", app=training())
    app.add_app(title="Cheat sheet", icon="training.webp", app=cheatsheet())
    app.add_app(title="Run a model", icon="analytics.ico", app=indivmodels())
    app.add_app(title="Automated analytics flow", icon="analytics.ico", app=autoanalytics())
    app.add_app(title="Custom data environment", icon="training.webp", app=customdataenv())
    app.add_app(title="Planning and forecasting", icon="analytics.ico", app=forecasting())
    app.add_loader_app(apps.MyLoadingApp(delay=0))

    st.session_state.allow_access=0

    complex_nav = {
            'Heuristix home': ['Heuristix home'],
            'ABI Analytics home': ['ABI Analytics home'],
            'Training': ["Training", "Cheat sheet"],
            'Run a model': ['Run a model'],
            'Automated analytics': ['Automated analytics flow'],
            'Custom data environment': ['Custom data environment'],
            'Planning and forecasting': ['Planning and forecasting']
        }

    # if st.session_state.pagechoice == "auto analytics":
    #     # complex_nav = "Automated analytics flow"
    #     pass

    #run the whole lot
    app.run(complex_nav)
