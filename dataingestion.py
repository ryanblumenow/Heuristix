import pandas as pd
import streamlit as st

import streamlit as st
import pandas as pd

# Cached function for expensive operations (e.g., loading default data)
@st.cache_data(experimental_allow_widgets=True)
def load_data(file_path):
    """Expensive operation to load data from a CSV file."""
    return pd.read_csv(file_path, low_memory=False)

# Main function to read data
def readdata():
    """Handles data ingestion with widget interactions."""
    # Initialize an empty dataframe
    df = None

    # Widget for file upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

    # Button to load default data
    defaultdata = st.button("Use default data")

    if uploaded_file is not None:
        # Load uploaded data
        df = pd.read_csv(uploaded_file)
        st.write("Dataset preview:")
        st.write(df.head())
    elif defaultdata:
        # Use cached function to load default dataset
        df = load_data("testdata.csv")
    else:
        st.warning("Please upload a dataset or use default data")

    return df


    ### End


