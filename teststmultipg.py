import streamlit as st
from click_bars import st_click_bars
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
from st_on_hover_tabs import on_hover_tabs
from heuristixhome import heuristix_home
from heuristixanalytix import analytix
from heuristixtraining import training
from heuristixrunamodel import run_a_model
from heuristixpredictions import predict
from heuristixalisen import alisen
import os
import subprocess

# st.set_page_config(layout="wide")

# Run dtale-streamlit as a subprocess
command = [
    "dtale-streamlit",
    "run",
    __file__,  # Use the current file
    "--theme.primaryColor=#FFCC66",
    "--client.showErrorDetails=false"
]

st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)

# Hide white space at top

hide_streamlit_style = """
    <style>
        #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 1rem;}
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
        #background-color: gold;
        padding: 5px;
        top: 2px;
        left: 630px;
    }</style>"""

st.markdown(customizedfooter, unsafe_allow_html=True)

with st.sidebar:
        st.image('Heuristix_icon.png')
        add_vertical_space(1)
        value = on_hover_tabs(tabName=['PRINCIPLES', 'ABOUT', 'CONTACT US', ''], 
                            iconName=['contacts', 'dashboard', 'account_tree', 'table', 'report', 'edit', 'update', 'pivot_table_chart', 'menu_book'],
                            styles = {'navtab': {'background-color': "#ffcc66", #'#6d0606',
                                                'color': 'black',
                                                'font-size': '18px',
                                                'transition': '.3s',
                                                'white-space': 'nowrap',
                                                'text-transform': 'uppercase',
                                                'font-weight': 'bold'},
                                    'tabOptionsStyle': {':hover :hover': {'color': '#dcac54',
                                                                    'cursor': 'pointer'}},
                                    'iconStyle':{'position':'fixed',
                                                    'left':'7.5px',
                                                    'text-align': 'left'},
                                    'tabStyle' : {'list-style-type': 'none',
                                                    'margin-bottom': '30px',
                                                    'padding-left': '30px'}},
                            key="hoversidebar",
                            default_choice=3)
        
        # st.text_area("", "Why Heuristix?\n\nHumans use HEURISTICS, scientific generalizations, to make decisions, solve problems, and form judgments quickly. They are simple strategies that can be used by humans, animals, organizations, and machines. Heuristic methods and general heuristical conclusions from data analysis can be used to speed up the process of finding a satisfactory solution and decising how to act. Heuristics are mental shortcuts that ease the cognitive load of making a decision, with the goal of making decisions more quickly, frugally, and/or accurately. This is the nexus of behavioral economics and data science, with artificial intelligence. Our Heuristix Analytix methodology provides a revolutionary guided analytical experience that gives you the best insights into yuor data, and tells you how to react and act in order to get the best possible outcome, for anything you'd like to achieve.", height=321)
        # st.markdown('<a href="https://en.wikipedia.org/wiki/Heuristic" style="color: white;">See more</a>', unsafe_allow_html=True)

        if value == "PRINCIPLES":
            st.write("")
            st.header("Heuristix, Augmented by Alisen")
            st.text_area("Our principles:", "1. We help you identify useful heuristics.\n2. We provide nuggets of analytical insight.\n3. We provide bespoke, guided artificial intelligence and prompts.\n\nIn a nutshell: adaptable heuristics, insight nuggets, prescriptive and bespoke AI engineering.", height=240)

        if value == "ABOUT":
            st.write("")
            st.header("What is Heuristix?")
            st.text_area(
                "",
                "Why Heuristix?\n\nHumans use HEURISTICS, scientific generalizations, to make decisions, solve problems, and form judgments quickly. They are simple strategies that can be used by humans, animals, organizations, and machines. Heuristic methods and general heuristical conclusions from data analysis can be used to speed up the process of finding a satisfactory solution and deciding how to act. Heuristics are mental shortcuts that ease the cognitive load of making a decision, with the goal of making decisions more quickly, frugally, and/or accurately. This is the nexus of behavioral economics and data science, with artificial intelligence. Our Heuristix Analytix methodology provides a revolutionary guided analytical experience that gives you the best insights into your data, and tells you how to react and act in order to get the best possible outcome, for anything you'd like to achieve.",
                height=321,
            )
        elif value == "CONTACT US":
            st.write("")
            # Custom HTML for Contact Page
            contact_html = """
            <div style="text-align: left; padding: 20px; font-family: Arial, sans-serif; line-height: 1.8;">
                <h2 style="color: black; font-weight: bold;">Contact Us</h2>
                <p style="font-size: 18px; color: black;">Please feel free to reach out:</p>
                <div style="margin: 0px 0;">
                    <div style="margin-bottom: 10px;">
                        <i class="fa fa-user" style="font-size: 21px; color: black;"></i>
                        <span style="font-size: 18px; margin-left: 10px; color: black;">Ryan Blumenow</span>
                    </div>
                    <div style="margin-bottom: 10px;">
                        <i class="fa fa-envelope" style="font-size: 21px; color: black;"></i>
                        <a href="mailto:blumenowster@gmail.com" style="font-size: 18px; margin-left: 10px; color: black; text-decoration: none;">
                            blumenowster@gmail.com
                        </a>
                    </div>
                    <div style="margin-bottom: 10px;">
                        <i class="fa fa-phone" style="font-size: 21px; color: black;"></i>
                        <a href="tel:+27769813490" style="font-size: 18px; margin-left: 10px; color: black; text-decoration: none;">
                            +27 (0) 76 981 3490
                        </a>
                    </div>
                </div>
            </div>
            """
            st.markdown(contact_html, unsafe_allow_html=True)

            # Add Font Awesome
            st.markdown(
                """<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet">""",
                unsafe_allow_html=True,
            )

css = '''
<style>
    .stTabs [data-baseweb="tab-highlight"] {
        background-color:blue;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

# Initialize Session State for Navigation and Click Handling
if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "Heuristix home"
if "last_clicked_bar" not in st.session_state:
    st.session_state["last_clicked_bar"] = None


# Custom Navbar using streamlit-option-menu
def custom_navbar():

    selected = option_menu(
        menu_title=None,  # required
        options=["Heuristix home", "Heuristix Analytix", "Training", "Make a prediction", "Ask Alisen"],  # required
        icons=["house", "bar-chart", "book", "play", "lightbulb"],  # optional
        menu_icon="cast",  # optional
        default_index=["Heuristix home", "Heuristix Analytix", "Training", "Make a prediction", "Ask Alisen"].index(
            st.session_state["selected_page"]
        ),  # Sync default index with the session state
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0 10px",
                "background-color": "#ffcc66", # "#800000",
                "border-radius": "10px",
                "display": "flex",
                "justify-content": "flex-start",  # Align items to the left
                "margin-left": "10px",  # Shift the navbar to the left
                "margin-right": "20px",
                "width": "100%",  # Optional: Adjust the navbar width
                "box-sizing": "border-box",  # Ensure padding and border are included in the total width
            },
            "icon": {
                "font-size": "18px",
            },
            "nav-link": {
                "font-size": "14px",
                "text-align": "center",
                "margin": "0 5px",  # Reduced margin to make it compact
                "color": "black",
                "padding": "10px",
                "border-radius": "10px",
                "--hover-color": " #ffd700"
            },
            "nav-link-selected": {
                "background-color": "white",
                "color": "#ffcc66",
                "border-radius": "8px",
            },
        },
    )

    # Inject custom CSS for selected icon styling
    st.markdown("""
        <style>
        .nav-link-selected svg {
            fill: #ffcc66 !important;  /* Change icon color to red for selected tab */
        }
        </style>
    """, unsafe_allow_html=True)

    if selected != st.session_state["selected_page"]:
        st.session_state["selected_page"] = selected
        st.rerun()

# # Custom Bars Component Logic
# def custom_bars():
#     # st.markdown("<h3 style='text-align: center;'>Click a Bar to Navigate</h3>", unsafe_allow_html=True)

#     # Embed the custom bars component
#     clicked_bar = st_click_bars()  # This is your custom component

#     # Handle bar click
#     if clicked_bar and clicked_bar != st.session_state["last_clicked_bar"]:
#         st.session_state["last_clicked_bar"] = clicked_bar
#         if clicked_bar == "Bar1":
#             st.session_state["selected_page"] = "Heuristix home"
#         elif clicked_bar == "Bar2":
#             st.session_state["selected_page"] = "ABI Analytics home"
#         elif clicked_bar == "Bar3":
#             st.session_state["selected_page"] = "Training"
#         elif clicked_bar == "Bar4":
#             st.session_state["selected_page"] = "Run a model"
#         st.rerun()  # Force reload only once


# Page Content Functions
# def heuristix_home():
#     st.title("Heuristix Home")
#     st.write("Welcome to the Heuristix Home Page.")


# def abi_analytics_home():
#     st.title("ABI Analytics Home")
#     st.write("Welcome to the ABI Analytics Home Page.")


# def training():
#     st.title("Training")
#     st.write("Welcome to the Training Page.")


# def run_a_model():
#     st.title("Run a Model")
#     st.write("Welcome to the Run a Model Page.")


# Main Page Rendering
def main():
    # Render Navbar
    custom_navbar()

    # Render Content Based on Selected Page
    if st.session_state["selected_page"] == "Heuristix home":
        heuristix_home()
    elif st.session_state["selected_page"] == "Heuristix Analytix":
        analytix()
    elif st.session_state["selected_page"] == "Training":
        training()
    elif st.session_state["selected_page"] == "Make a prediction":
        predict()
    elif st.session_state["selected_page"] == "Ask Alisen":
        alisen()

    # # Render Custom Bars
    # custom_bars()
    
# Run the App
if __name__ == "__main__":
    main()