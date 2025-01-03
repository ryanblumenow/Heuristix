# from pandasai.llm.local_llm import LocalLLM
import streamlit as st
import pandas as pd
import pandasai
import matplotlib.pyplot as plt
import dataingestion
from streamlit_extras.add_vertical_space import add_vertical_space
from pandasai import Agent
from pandasai import SmartDataframe
import os
import pandasai.helpers.cache
import shelve
# pandasai.helpers.cache.Cache.__init__ = lambda self: setattr(self, 'cache', shelve.open('/tmp/pandasai_cache'))
# pandasai._cache = Cache(filename="pandasai_cache.db")
from pandasai.helpers.cache import Cache

# Override the Cache class to use an in-memory dictionary instead of shelve
Cache.__init__ = lambda self: setattr(self, 'cache', {})
Cache.get = lambda self, key: self.cache.get(key, None)
Cache.set = lambda self, key, value: self.cache.__setitem__(key, value)
Cache.clear = lambda self: self.cache.clear()

def alisen():

    st.markdown(f"""
        ### <span style="color: #ffb31a;">Alisen</span>: the <span style="color: #ffb31a;">A</span>rtificial <span style="color: #ffb31a;">L</span>earning and <span style="color: #ffb31a;">I</span>ntelligence <span style="color: #ffb31a;">S</span>ynthesis <span style="color: #ffb31a;">En</span>gine
        """, unsafe_allow_html=True)

    st.text_area("Heuristix's Alisen - helping you ask better questions and make better decisions", "A virtual advisor, acting as your second-in-charge, helping with data analysis using custom Generative Artificial Cognition and Intelligence using AI Agents.\n\nBespoke to this environment's data and user prompts, and never leaving the environment i.e. trained on specific data and totally secure.")

    st.info("This is decision support: artifically intelligent advanced domain expertise and organizational knowledge. Alisen and her agents are trained on and sourced from your data, so they are experts in your specific environment.")

    # Initialize the local model
    model = LocalLLM(
        api_base="http://localhost:11434/v1",
        model="llama3"
    )

    os.environ["PANDASAI_API_KEY"] = "$2a$10$H2DDxvj8BynWr.Tsk1Glb.ci4R5NUiVfdRrMcrldkwpHzp5HEZrz2"

    df = dataingestion.readdata()

    # df = SmartDataframe(df_raw)

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
    # agent = Agent(df, config={"llm": model})

    agent = Agent(df)

    alisencolm1, alisencolm2, alisencolm3 = st.columns([1,3,1])

    with alisencolm2:
        alisen = st.image('./gui/images/alisenillus.png', width=600)

    st.info("Hello! I'm Alisen. My role is to help you understand the data I have stored in my memory, and decide what actions might be best to take based on the insights I can offer.\n\nI, and my team of AI agents, am here to augment your experience with our expertise, in line with our three guiding principles: heuristics based on understanding the data, insightful nuggets based on advanced analytics and deep dives into the daata, and augmentation with bespoke artificial cognition and intelligence. That's where I come in. I'm here to help!")

    # Prompt input
    prompt = st.text_input("What can I help you with today? Enter a prompt for my analysis here. Be as specific as possible for best results.")

    print(prompt)

    if st.button("Ask Alisen"):
        if prompt:
            with st.spinner("I'm working with my team after gathering the context. Please bear with me for a few moments."):
                try:
                    # Generate response
                    response = agent.chat(prompt)
                    explanation = agent.explain()
                    st.write(response)
                    st.write(explanation)
                    # Handle graphical outputs
                    if isinstance(plt.gcf(), plt.Figure):
                        # Display the current figure if a plot was generated
                        st.pyplot(plt.gcf())
                        plt.clf()  # Clear the figure to avoid overlapping
                    else:
                        # Display non-graphical outputs
                        st.write(response)
                        st.write(explanation)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
