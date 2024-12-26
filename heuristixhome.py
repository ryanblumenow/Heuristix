import streamlit as st
from click_bars import st_click_bars

def heuristix_home():
    st.title("Heuristix Home")
    st.write("Welcome to the Heuristix Home Page.")

    # Custom Bars Component Logic
    def custom_bars():
        # st.markdown("<h3 style='text-align: center;'>Click a Bar to Navigate</h3>", unsafe_allow_html=True)

        # Embed the custom bars component
        clicked_bar = st_click_bars()  # This is your custom component

        # Handle bar click
        if clicked_bar and clicked_bar != st.session_state["last_clicked_bar"]:
            st.session_state["last_clicked_bar"] = clicked_bar
            if clicked_bar == "Bar1":
                st.session_state["selected_page"] = "Heuristix Analytix"
            elif clicked_bar == "Bar2":
                st.session_state["selected_page"] = "Training"
            elif clicked_bar == "Bar3":
                st.session_state["selected_page"] = "Make a prediction"
            elif clicked_bar == "Bar4":
                st.session_state["selected_page"] = ""
            st.experimental_rerun()  # Force reload only once

    # Render Custom Bars
    custom_bars()

