import streamlit as st
from click_handler_component import click_handler

# Test the click handler component
clicked_bar = click_handler()

# Display the result
st.write("Bar Clicked:", clicked_bar)

# Navigate based on the clicked bar
if clicked_bar == "Dwell":
    st.write("Navigating to Dwell...")
elif clicked_bar == "Dwell 2":
    st.write("Navigating to Dwell 2...")
elif clicked_bar == "Artificial Intelligence":
    st.write("Navigating to Artificial Intelligence...")
