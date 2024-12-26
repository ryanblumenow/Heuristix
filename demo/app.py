import streamlit as st
from click_image import st_click_image
from st_click_detector import click_detector

value = st_click_image()
if value is None:
    st.stop()

st.write("Received", value)

col1, col2, col3 = st.columns([1,2,1])

with col1:
    pass

with col2:
    if value=='Image1':

        content2 = """
        <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
            <style>
            .zoom {
            padding: 10px;
            width: 290px;
            height: 210px;
            transition: transform .21s; /* Animation */
            margin: 0 auto;
            }
            .zoom:hover {
            transform: scale(1.1); /* (110% zoom - Note: if the zoom is too large, it will go outside of the viewport) */
            }
            </style>
            <div class="w3-container">
            <a href='#' id='Image 1'><img width='21%' src='https://i.postimg.cc/SK23bN61/analytics.jpg' class='zoom w3-round-xxlarge w3-hover-opacity'></a>
            <div class="w3-display-middle w3-large"><b><h3>Get started!</h3></b></div>
            </div>
            """

        clicked = click_detector(content2)

        st.markdown(f"**{clicked} clicked**" if clicked != "" else "**No click**")

    else:

        st.write('No picture here')

with col3:
    pass

