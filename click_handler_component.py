import os
import streamlit.components.v1 as components

# Declare the custom component
click_handler = components.declare_component(
    "click_handler", path=os.path.join(os.path.dirname(__file__), "click_handler_component", "build")
)
