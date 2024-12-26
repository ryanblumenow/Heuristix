import streamlit as st
# from streamlit.state.session_state import SessionState
# Custom imports 
from multipage2abi import MultiPage
#from multipage import save, MultiPage, start_app, clear_cache
import mainui # import your pages here

from PIL import Image

from functions import *

# Create an instance of the app 
abidashboard = MultiPage()

ico = Functions.set_svg_icon('martechico.ico')

image=Image.open(ico)

st.set_page_config(layout="wide",page_title='ABI Analytics', page_icon=image)

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #ff3333;">
  <a class="navbar-brand" href="" target="_blank">ABI analytics</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item">
        <a class="nav-link" href="www.google.com" target="_blank">ABI analytics objectives</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="www.google.com" target="_blank">ABI analytics roadmap</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="www.google.com" target="_blank">Help</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)

# SessionState.selectedsite = 'Dagara'

# Add all your applications (pages) here
abidashboard.add_page("Main dashboard interface", mainui.abiui)

# The main app
abidashboard.run()