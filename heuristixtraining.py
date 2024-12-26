import streamlit as st
from streamlit_option_menu import option_menu
import os, base64

def heuristixtraining():

    title = '<p style="font-family:sans-serif; color:gold; font-size: 39px; text-align: center;"><b>How to use the Heuristix Analytix Playground</b></p>'
    st.markdown(title, unsafe_allow_html=True)

    st.session_state['pagechoice'] = 'training'

    st.header("How do you use our methodology?")

    nav = st.container()
    modelsec = st.container()
    results = st.container()

    with nav:

        with st.expander("Navigating the Heuristix Playground"):
            st.write("This is intended to be an automated 2IC (Second in Charge) - a knowledgable advisor who guides you on what insights you can glean from data, and what decisions should be made using that data. Follow the automated Heuristix Analytix flow if you want to see live data insights in action - even if you aren't a data scientist! This is a fully automated experience that guides you through each step of the analytics process, tells you what results you get from each step and how they relate to each other, and gives you real-time advice on how to actually use those insights. Drop us a line for a guided demonstration or to see how to partner with us.")
            st.write("This system is intended to achieve two things, with your (or our) data: to help you ask better questions, and to make better decisions. Use either the default data (on education and earnings, alcohol choices, or financial product choice) as demonstrators, or load up your own proprietary dataset (we do not store this) and see live insights on your own data!")
            st.info("Imagine what you could do with this capability, in-house, tailored to your own environment, with a live advisor trained on your data and domain expertise. That's the very definition of a competitive advantage.", icon="ℹ️")

    with modelsec:

        with st.expander("The models that have been included"):
            st.metric(label="Number of models", value=10, delta="10")
            st.write("")
            modelsincluded = ["Exploratory data analysis", "Correlation analysis", "Hypothesis testing", "Dimension reduction (using random forest ensemble method)", "ANOVA analysis", "Linear Regression", "Cluster analysis (K-means)", "Conjoint analysis", "Neural networks", "Decision trees", "Predictive analytics and genAI", ]
            st.write(modelsincluded)

    with results:
        # st.write("[Interpreting these models](https://analyticsindiamag.com/10-machine-learning-algorithms-every-data-scientist-know/)")
        pdf_path = os.path.abspath("./Training/HeuristixTrainingGuide.pdf")
        with st.expander("Interpreting these models"):
            with open(pdf_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                # Embedding PDF in HTML
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="1500" height="1200" type="application/pdf"></iframe>'
                # Displaying File
                st.markdown(pdf_display, unsafe_allow_html=True)

    