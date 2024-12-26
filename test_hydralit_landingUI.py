from hydralit import HydraApp, HydraHeadApp
import streamlit as st
from click_handler_component import click_handler

# Define the Home App
class HomeApp(HydraHeadApp):
    def __init__(self, title="Home", **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):
        try:
            st.info("Welcome to the Home Page!")

            # Render the custom component
            clicked_bar = click_handler()
            
            st.write(f"Bar Clicked: {clicked_bar}")

            # Redirect to selected page
            if clicked_bar:
                self.do_redirect(clicked_bar)

        except Exception as e:
            st.write(f"Raw clicked_bar value: {clicked_bar}")
            st.error(f"An error occurred: {e}")


# Define the Dwell App
class DwellApp(HydraHeadApp):
    def __init__(self, title="Dwell", **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):
        st.title("Dwell Page")
        st.markdown("Welcome to the Dwell Page!")


# Define the Dwell 2 App
class Dwell2App(HydraHeadApp):
    def __init__(self, title="Dwell 2", **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):
        st.title("Dwell 2 Page")
        st.markdown("Welcome to the Dwell 2 Page!")


# Define the AI App
class AIApp(HydraHeadApp):
    def __init__(self, title="AI", **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):
        st.title("Artificial Intelligence Page")
        st.markdown("Welcome to the AI Page!")


# Main Hydralit App
if __name__ == "__main__":
    app = HydraApp(title="Hydralit Bar Chart Navigation")

    # Add apps
    app.add_app("Home", app=HomeApp(title="Home"), is_home=True)
    app.add_app("Dwell", app=DwellApp(title="Dwell"))
    app.add_app("Dwell 2", app=Dwell2App(title="Dwell 2"))
    app.add_app("Artificial Intelligence", app=AIApp(title="AI"))

    # Run the app
    app.run()
