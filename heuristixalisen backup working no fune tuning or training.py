from pandasai.llm.local_llm import LocalLLM
import streamlit as st
import pandas as pd
from pandasai import Agent
import matplotlib.pyplot as plt
import dataingestion
from streamlit_extras.add_vertical_space import add_vertical_space

def alisen():

    st.markdown(f"""
        ### <span style="color: #ffb31a;">Alisen</span>: the <span style="color: #ffb31a;">A</span>rtificial <span style="color: #ffb31a;">L</span>earning and <span style="color: #ffb31a;">I</span>ntelligence <span style="color: #ffb31a;">S</span>ynthesis <span style="color: #ffb31a;">En</span>gine
        """, unsafe_allow_html=True)

    st.text_area("Heuristix's Alisen - helping you ask better questions and make better decisions", "A virtual advisor, acting as your second-in-charge, helping with data analysis using custom Generative Artificial Cognition and Intelligence using AI Agents.\n\nBespoke to this environment's data and user prompts, and never leaving the environment i.e. trained on specific data and totally secure.")

    st.info("This is decision support: artifically intelligent advanced domain expertise and organizational knowledge. Alisen and her agents are trained on and sourced from your data, so they are experts in your specific environment.")

    # Initialize the local model
    model = LocalLLM(
        api_base="http://localhost:11434/v1",
        model="llama3")

    df = dataingestion.readdata()

    with st.expander("Sample of data", expanded=False):
        samplecol1, samplecol2, samplecol3 = st.columns([1,3,1])
        with samplecol2:
            st.write(df.sample(8))

    with st.expander("What can Alisen do?", expanded=False):
        alisencol1, alisencol2, alisencol3 = st.columns([1,3,1])
        with alisencol2:
            st.image("howdoesalisenwork.png")

    add_vertical_space(3)
            
    # Initialize the agent
    agent = Agent(df, config={"llm": model})

    alisencolm1, alisencolm2, alisencolm3 = st.columns([1,3,1])

    with alisencolm2:
        alisen = st.image('./gui/images/alisenillus.jpg', width=600)

    st.info("Hello! I'm Alisen. My role is to help you understand the data I have stored in my memory, and decide what actions might be best to take based on the insights I can offer.\n\nI and my team of AI agents are here to augment your experience with our expertise, in line with our three guiding principles: heuristics based on understanding the data, insightful nuggets based on advanced analytics and deep dives into the daata, and augmentation with bespoke artificial cognition and intelligence. That's where I come in. I'm here to help!")

    # Prompt input
    prompt = st.text_input("What can I help you with today? Enter a prompt for my analysis here. Be as specific as possible for best results.")

    if st.button("Ask Alisen"):
        if prompt:
            with st.spinner("Generating response..."):
                try:
                    # Generate response
                    response = agent.chat(prompt)

                    # Handle graphical outputs
                    if isinstance(plt.gcf(), plt.Figure):
                        # Display the current figure if a plot was generated
                        st.pyplot(plt.gcf())
                        plt.clf()  # Clear the figure to avoid overlapping
                    else:
                        # Display non-graphical outputs
                        st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
