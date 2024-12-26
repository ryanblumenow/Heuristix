import streamlit as st
from streamlit_option_menu import option_menu

#add an import to Hydralit
from hydralit import HydraHeadApp

#create a wrapper class
class training(HydraHeadApp):

#wrap all your code in this method and you should be done
    def run(self):
        #-------------------existing untouched code------------------------------------------
        
        # st.title('Small Application with a table and chart.')

        title = '<p style="font-family:sans-serif; color:red; font-size: 39px; text-align: center;"><b>How to use the Analytics Playground</b></p>'
        st.markdown(title, unsafe_allow_html=True)

        st.session_state['pagechoice'] = 'training'

        with st.sidebar:
            choose = option_menu("ABI Analytics", ["Keep in mind", "Show me the steps", "Give me tips", "Give feedback"],
                                icons=['key', 'bezier2', 'joystick', 'keyboard'],
                                menu_icon="app-indicator", default_index=0,
                                styles={
                "container": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"color": "black", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#ab0202"},
            }
            )

            print(st.session_state.pagechoice)

            if choose == "Keep in mind":
                st.write("Remember to ask good questions. That is the basis of making good decisions.")

            if choose == "Show me the steps" and st.session_state.pagechoice=="test":
                st.write("Steps you should follow:")

            if choose == "Give me tips":
                st.write("Here are some tips:")

            if choose == "Give feedback":
                st.write("Give feedback")
                with st.sidebar.form(key='columns_in_form',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
                    rating=st.slider("Please rate the app", min_value=1, max_value=5, value=3,help='Drag the slider to rate the app. This is a 1-5 rating scale where 5 is the highest rating')
                    text=st.text_input(label='Please leave your feedback here')
                    submitted = st.form_submit_button('Submit')
                    if submitted:
                        st.write('Thanks for your feedback!')
                        st.markdown('Your Rating:')
                        st.markdown(rating)
                        st.markdown('Your Feedback:')
                        st.markdown(text)

        st.header("Make these proprietarily")

        nav = st.container()
        modelsec = st.container()
        results = st.container()

        with nav:

            with st.expander("Navigating the Playground"):
                st.write("Navigation parameters")

        with modelsec:

            with st.expander("The models that have been included"):
                st.metric(label="Number of models", value=10, delta="10")
                st.write("")
                modelsincluded = ["Hypothesis Testing", "Linear Regression", "Logistic Regression", "Clustering", "ANOVA", "Principal Component Analysis", "Conjoint Analysis", "Neural Networks", "Decision Trees", "Ensemble Methods"]
                st.write(modelsincluded)

        with results:
            st.write("[Interpreting these models](https://analyticsindiamag.com/10-machine-learning-algorithms-every-data-scientist-know/)")

