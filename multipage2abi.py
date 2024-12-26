# This file is the framework for generating multiple Streamlit applications through an object oriented framework.

# Import necessary libraries 
from mainui import abiui
import streamlit as st
# from streamlit.state.session_state import SessionState
from PIL import Image
from functions import *
import keyboard
import os
import signal
import streamlit.components.v1 as components

# Define the multipage class to manage the multiple apps in our program 

class MultiPage: 
    # Framework for combining multiple streamlit applications.

    def __init__(self) -> None:
        # Constructor class to generate a list which will store all our applications as an instance variable.
        self.pages = []
    
    def add_page(self, title, func) -> None: 
        # Class Method to Add pages to the project

        # Args:
        #     title ([str]): The title of page which we are adding to the list of apps 
            
        #     func: Python function to render this page in Streamlit

        self.pages.append({
          
                "title": title, 
                "function": func
            })

    def run(self):
        # Dropdown to select the page to run

        html = """
        <style>
            .reportview-container {
            flex-direction: row-reverse;
            }

            header > .toolbar {
            flex-direction: row-reverse;
            left: 1rem;
            right: auto;
            }

            .sidebar .sidebar-collapse-control,
            .sidebar.--collapsed .sidebar-collapse-control {
            left: auto;
            right: 0.5rem;
            }

            .sidebar .sidebar-content {
            transition: margin-right .3s, box-shadow .3s;
            }

            .sidebar.--collapsed .sidebar-content {
            margin-left: auto;
            margin-right: -21rem;
            }

            @media (max-width: 991.98px) {
            .sidebar .sidebar-content {
                margin-left: auto;
            }
            }
        </style>
        """
        # st.markdown(html, unsafe_allow_html=True) # This line and the html above are for the right-hand sidebar

        side_bar = """
        <style>
            /* The whole sidebar */
            .css-1lcbmhc.e1fqkh3o0{
            margin-top: 3.8rem;
            }
            
            /* The display arrow */
            .css-sg054d.e1fqkh3o3 {
            margin-top: 5rem;
            }

            /* The display arrow */
            .css-bauj2f.e1fqkh3o3 {
            margin-top: 5rem;
            }

        </style> 
        """
        st.markdown(side_bar, unsafe_allow_html=True) # This moves the sidebar down to accommodate the navigation bar at the top

        # st.title("New Sidebar")
        st.sidebar.text("ABI functions")

        preheader = st.container()
        header1 = st.container()

        with preheader:
            components.html('''
            <!DOCTYPE html>
            <html>

            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons">

                <style>
                    body {
                        font-family: "Lato", sans-serif;
                    }
                    
                    .sidebar {
                        height: 100%;
                        width: 39px;
                        position: fixed;
                        z-index: 1;
                        top: 0;
                        left: 0;
                        background-color: #A7A9AA;
                        overflow-x: hidden;
                        transition: 0.5s;
                        padding-top: 60px;
                        white-space: nowrap;
                    }
                    
                    .sidebar a {
                        padding: 1px 1px 1px 8px;
                        text-decoration: none;
                        font-size: 18px;
                        color: #818181;
                        display: block;
                        transition: 0.3s;
                    }
                    
                    .sidebar a:hover {
                        color: #f1f1f1;
                    }
                    
                    .sidebar .closebtn {
                        position: absolute;
                        top: 0;
                        right: 14px;
                        font-size: 14px;
                        margin-left: 5px;
                    }
                    
                    .material-icons,
                    .icon-text {
                        vertical-align: middle;
                    }
                    
                    .material-icons {
                        padding-bottom: 1px;
                    }
                    
                    #main {
                        transition: margin-left .5s;
                        padding: 19px;
                        margin-left: 39px;
                    }
                    /* On smaller screens, where height is less than 450px, change the style of the sidenav (less padding and a smaller font size) */
                    
                    @media screen and (max-height: 450px) {
                        .sidebar {
                            padding-top: 5px;
                        }
                        .sidebar a {
                            font-size: 14px;
                        }
                    }
                </style>
            </head>

            <body>

                <div id="mySidebar" class="sidebar" onmouseover="toggleSidebar()" onmouseout="toggleSidebar()">
                    <a href="#"><span><i class="material-icons">info</i><span class="icon-text">&nbsp;&nbsp;&nbsp;&nbsp;About</span></a><br>
                    <a href="#"><i class="material-icons">spa</i><span class="icon-text"></span>&nbsp;&nbsp;&nbsp;&nbsp;Services</a></span>
                    </a><br>
                    <a href="#"><i class="material-icons">email</i><span class="icon-text"></span>&nbsp;&nbsp;&nbsp;&nbsp;Contact<span></a>
                </div>

                <div id="main">
                    <h2></h2>
                    <p></p>
                    <p></p>
                </div>

                <script>
                    var mini = true;

                    function toggleSidebar() {
                        if (mini) {
                            console.log("opening sidebar");
                            document.getElementById("mySidebar").style.width = "250px";
                            document.getElementById("main").style.marginLeft = "250px";
                            this.mini = false;
                        } else {
                            console.log("closing sidebar");
                            document.getElementById("mySidebar").style.width = "39px";
                            document.getElementById("main").style.marginLeft = "39px";
                            this.mini = true;
                        }
                    }
                </script>

            </body>

            </html>''')

        with header1:
            clm1, clm2, clm3, clm4, clm5 = st.columns(5)
            with clm1:
                pass
            with clm2:
                pass
            with clm3:
                image1 = Functions.set_svg_image('Martechlogo.jpg')
                image2 = Image.open(image1)
                st.image(image2, width=283)
            with clm4:
                pass
            with clm5:
                quitapp = st.button("Exit", key="multipgexit")
        
        if quitapp==True:
            keyboard.press_and_release('ctrl+w')
            os.kill(os.getpid(), signal.SIGTERM)

        # else:
        seloption = st.selectbox('ABI Analytics Navigation', ['Main ABI page'], key='foo') # format_func=lambda page: page['title'],
        if seloption == 'Main ABI page':
            page = {'title': 'Main dashboard interface', 'function' : abiui}
        
        # run the selected app function

        def handle_click():
            st.session_state.foo = 'Main ABI page'

        mainbtn = st.sidebar.button("Main ABI page", on_click=handle_click)
        
        if mainbtn == True:
            page={'title': 'Main ABI page', 'function': abiui}
            

        page['function']()
        


