import streamlit as st
from bokeh.models.widgets import Div

#add an import to Hydralit
from hydralit import HydraHeadApp

#create a wrapper class
class cheatsheet(HydraHeadApp):

#wrap all your code in this method and you should be done
    def run(self):
        #-------------------existing untouched code------------------------------------------
        
        # st.title('Cheat sheets and guides')

        title = '<p style="font-family:sans-serif; color:red; font-size: 39px; text-align: center;"><b>Cheat sheets and guides</b></p>'
        st.markdown(title, unsafe_allow_html=True)

        st.session_state['pagechoice'] = 'cheat sheets'

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

        st.write("")

        st.header("Background material")

        col1, col2, col3 = st.columns(3)

        with col1:
            pass
        with col2:
            pass
        with col3:
            pass

        st.write("")

        col1, col2, col3 = st.columns(3)

        with col1:

            if st.button('Understanding advanced analytics'):
                js = "window.open('https://blogs.sas.com/content/subconsciousmusings/2020/12/09/machine-learning-algorithm-use/')"  # New tab or window
                # js = "window.location.href = 'https://www.streamlit.io/'"  # Current tab
                html = '<img src onerror="{}">'.format(js)
                div = Div(text=html)
                st.bokeh_chart(div)

        with col2:

            if st.button('Understanding machine learning algorithms'):
                js = "window.open('https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-cheat-sheet')"  # New tab or window
                # js = "window.location.href = 'https://www.streamlit.io/'"  # Current tab
                html = '<img src onerror="{}">'.format(js)
                div = Div(text=html)
                st.bokeh_chart(div)

        with col3:

            if st.button('How to select an appropriate algorithm'):
                js = "window.open('https://docs.microsoft.com/en-us/azure/machine-learning/how-to-select-algorithms')"  # New tab or window
                # js = "window.location.href = 'https://www.streamlit.io/'"  # Current tab
                html = '<img src onerror="{}">'.format(js)
                div = Div(text=html)
                st.bokeh_chart(div)

        st.write("")

        st.header("Guided selection of estimation models")

        col1, col2, col3 = st.columns(3)

        with col1:
            pass
        with col2:
            pass
        with col3:
            pass
        
        st.write("")

        cl1, cl2, cl3 = st.columns(3)
        with cl1:
            pass
        with cl2:
            if st.button('How to choose the right estimator (with background material)'):
                js = "window.open('https://scikit-learn.org/stable/tutorial/machine_learning_map/#')"  # New tab or window
                # js = "window.location.href = 'https://www.streamlit.io/'"  # Current tab
                html = '<img src onerror="{}">'.format(js)
                div = Div(text=html)
                st.bokeh_chart(div)
        with cl3:
            pass

        st.write("")

        st.header("Understanding the process")

        col1, col2, col3 = st.columns(3)

        with col1:
            pass
        with col2:
            pass
        with col3:
            pass
        
        st.write("")

        st.image("datascience.jpg")

        st.write("")

        st.image("explainableai.png")

        st.write("")

        st.image("mlcheatsheet.png")

