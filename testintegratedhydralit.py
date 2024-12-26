import streamlit as st
from hydralit import HydraApp, HydraHeadApp
from click_bars import st_click_bars  # Assuming this is your custom component

class MainApp(HydraHeadApp):
    def run(self):
        st.title("Welcome to the Main App")
        clicked_bar = st_click_bars()  # Capture the clicked bar ID
        st.write(f"Clicked Bar: {clicked_bar}")  # Debugging
        
        # Handle redirection based on bar clicks
        if clicked_bar == "Bar1":
            st.session_state['selected_app'] = "Heuristix home"
            self.do_redirect("Heuristix home")
        elif clicked_bar == "Bar2":
            st.session_state['selected_app'] = "ABI Analytics home"
            self.do_redirect("ABI Analytics home")
        elif clicked_bar == "Bar3":
            st.session_state['selected_app'] = "Automated analytics flow"
            self.do_redirect("Automated analytics flow")
        else:
            st.info("Click on a bar to navigate.")

class HeuristixHome(HydraHeadApp):
    def run(self):
        st.title("Heuristix Home")
        st.write("Welcome to the Heuristix Home Page.")

class ABIAnalyticsHome(HydraHeadApp):
    def run(self):
        st.title("ABI Analytics Home")
        st.write("Welcome to the ABI Analytics Home Page.")

class AutomatedAnalyticsFlow(HydraHeadApp):
    def run(self):
        st.title("Automated Analytics Flow")
        st.write("Welcome to the Automated Analytics Flow Page.")

if __name__ == "__main__":
    # Initialize the HydraApp
    app = HydraApp(
        title="Consolidated Hydralit App",
        favicon="favicon.ico",
        nav_horizontal=True,
        navbar_sticky=True,
    )
    
    # Add apps
    app.add_app("Main App", MainApp())
    app.add_app("Heuristix home", HeuristixHome())
    app.add_app("ABI Analytics home", ABIAnalyticsHome())
    app.add_app("Automated analytics flow", AutomatedAnalyticsFlow())

    # Run the app
    app.run()
