import streamlit as st
# from streamlit.state.session_state import SessionState
from PIL import Image
from gui.uis.windows.main_window.functions_main_window import *
import keyboard
import os
import signal

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

        header1 = st.container()

        with header1:
            clm1, clm2, clm3, clm4, clm5 = st.columns(5)
            with clm1:
                pass
            with clm2:
                pass
            with clm3:
                image1 = Functions.set_svg_image('GSLogo.jpg')
                image2 = Image.open(image1)
                st.image(image2, width=283)
            with clm4:
                pass
            with clm5:
                quitapp = st.button("Exit")
        
        if quitapp==True:
            keyboard.press_and_release('ctrl+w')
            os.kill(os.getpid(), signal.SIGTERM)