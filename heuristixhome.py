import streamlit as st
from click_bars import st_click_bars
from streamlit_carousel import carousel
import streamlit.components.v1 as components

def heuristix_home():
    st.title("Heuristix Home")
    st.write("Welcome to the Heuristix home page, and our three pillars.")

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
            st.rerun()  # Force reload only once

    # Render Custom Bars
    custom_bars()

    # # Define carousel items with 'img', 'title', and 'text'
    # test_items = [
    #     dict(
    #         img="imagine.png",
    #         title="",
    #         text="Being able to know the right decision to make, every time",
    #     ),
    #     dict(
    #         img="imagine.png",
    #         title="",
    #         text="Your data as the backbone of a customized, automated, advanced analytics and AI cognitive system",
    #     ),
    #     dict(
    #         img="Imagine.png",
    #         title="",
    #         text="Knowing what training a client will find most valuable, what alcohol they prefer, what financial products they need...or any other predicted client behaviour, tailored for your product or service offering",
    #     ),
    # ]

    # # Render the carousel

    # st.write("")
    # st.write("")
    # st.write("")
    
    # carousel(
    #     items=test_items,
    #     container_height=290,  # Adjust to fit image and text
    #     width=0.5,             # Adjust width
    #     pause=True,            # Pause on hover
    #     wrap=True,             # Enable slide wrapping
    #     fade=True,             # Enable fade transition
    #     interval=5000          # Time interval between slides in ms
    # )

    components.html(
        """
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    * {box-sizing: border-box;}
    body {font-family: Verdana, sans-serif;}
    .mySlides {display: none;}
    img {vertical-align: middle;}

    /* Slideshow container */
    .slideshow-container {
    max-width: 600px;
    height: 300px;
    position: relative;
    margin: auto;
    }

    /* Caption text */
    .text {
    color: black; /* #f2f2f2; */
    font-size: 21px;
    padding: 8px 12px;
    position: absolute;
    bottom: 21px;
    width: 100%;
    text-align: center;
    }

    /* Number text (1/3 etc) */
    .numbertext {
    color: black; /* #f2f2f2; */
    font-size: 12px;
    padding: 8px 12px;
    position: absolute;
    top: 0;
    }

    /* The dots/bullets/indicators */
    .dot {
    height: 15px;
    width: 15px;
    margin: 0 2px;
    background-color: #bbb;
    border-radius: 50%;
    display: inline-block;
    transition: background-color 0.6s ease;
    }

    .active {
    background-color: #717171;
    }

    /* Fading animation */
    .fade {
    animation-name: fade;
    animation-duration: 1.5s;
    }

    @keyframes fade {
    from {opacity: .4} 
    to {opacity: 1}
    }

    /* On smaller screens, decrease text size */
    @media only screen and (max-width: 300px) {
    .text {font-size: 11px}
    }
    </style>
    </head>
    <body>

    <h2> </h2> <!-- Heading here -->
    <p> </p> <!-- Subheading here -->

    <div class="slideshow-container">

    <div class="mySlides fade">
    <div class="numbertext">1 / 5</div>
    <img src="https://i.postimg.cc/k5bBnBLj/imaginegold.png" style="width:100%">
    <div class="text" style="white-space: pre-wrap;">Being able to know the right decision to make, every time, with powerful decision support.</div>
    </div>

    <div class="mySlides fade">
    <div class="numbertext">2 / 5</div>
    <img src="https://i.postimg.cc/k5bBnBLj/imaginegold.png" style="width:100%">
    <div class="text" style="white-space: pre-wrap;">Your data as the backbone of a customized, automated, advanced analytics and AI cognitive system.</div>
    </div>

    <div class="mySlides fade">
    <div class="numbertext">3 / 5</div>
    <img src="https://i.postimg.cc/k5bBnBLj/imaginegold.png" style="width:100%">
    <div class="text" style="white-space: pre-wrap;">Knowing what training a client will find most valuable, what alcohol they prefer, what financial products they need...</div>
    </div>

    <div class="mySlides fade">
    <div class="numbertext">4 / 5</div>
    <img src="https://i.postimg.cc/k5bBnBLj/imaginegold.png" style="width:100%">
    <div class="text" style="white-space: pre-wrap;">...or any other predicted client behaviour, tailored for your product or service offering.</div>
    </div>

    <div class="mySlides fade">
    <div class="numbertext">5 / 5</div>
    <img src="https://i.postimg.cc/k5bBnBLj/imaginegold.png" style="width:100%">
    <div class="text" style="white-space: pre-wrap;">Welcome to Heuristix.</div>
    </div>

    </div>
    <br>

    <div style="text-align:center">
    <span class="dot"></span> 
    <span class="dot"></span> 
    <span class="dot"></span> 
    <span class="dot"></span> 
    <span class="dot"></span> 
    </div>

    <script>
    let slideIndex = 0;
    showSlides();

    function showSlides() {
    let i;
    let slides = document.getElementsByClassName("mySlides");
    let dots = document.getElementsByClassName("dot");
    for (i = 0; i < slides.length; i++) {
        slides[i].style.display = "none";  
    }
    slideIndex++;
    if (slideIndex > slides.length) {slideIndex = 1}    
    for (i = 0; i < dots.length; i++) {
        dots[i].className = dots[i].className.replace(" active", "");
    }
    slides[slideIndex-1].style.display = "block";  
    dots[slideIndex-1].className += " active";
    setTimeout(showSlides, 5000); // Change image every 5 seconds
    }
    </script>

    </body>
    </html> 

        """,
        height=600,
    )

