from pandasai.llm.local_llm import LocalLLM
import streamlit as st
import pandas as pd
from pandasai import Agent
import matplotlib.pyplot as plt
import dataingestion
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import os

# Custom Dataset class
def create_torch_dataset(tokenized_data):
    class CustomDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {
                key: torch.tensor(val[idx]) for key, val in self.encodings.items()
            }

    # Ensure labels are the same as input_ids for causal language modeling
    tokenized_data["labels"] = tokenized_data["input_ids"].clone()
    return CustomDataset(tokenized_data)

# Fine-tune the model
def fine_tune_model(data, output_dir="./fine_tuned_model"):
    with st.spinner("Fine-tuning the model on organizational data..."):
        tokenizer = AutoTokenizer.from_pretrained("./local_model", local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained("./local_model", local_files_only=True)

        # Add a padding token if necessary
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

        # Split data into individual samples (if data is a DataFrame, convert to a list of strings)
        if isinstance(data, pd.DataFrame):
            data = data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()
        elif isinstance(data, str):
            data = [data]

        # Tokenize the data
        tokenized_data = tokenizer(data, truncation=True, padding=True, return_tensors="pt")

        # Create a torch dataset
        train_dataset = create_torch_dataset(tokenized_data)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            num_train_epochs=3,
            save_steps=100,
            save_total_limit=2,
            logging_dir="./logs"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset
        )

        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        return output_dir

# Setup embedding index
def setup_embedding_index(documents, index_path="faiss_index"):
    with st.spinner("Loading domain knowledge into embedding index..."):
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(index_path)
        return vector_store

# Query embedding index
def query_embedding_index(query, vector_store):
    retriever = vector_store.as_retriever()
    return retriever.get_relevant_documents(query)

# Generate domain-specific prompts
def generate_domain_specific_prompt(prompt, retrieved_context):
    domain_context = "\n".join([doc.page_content for doc in retrieved_context])
    return f"Domain Context:\n{domain_context}\n\nUser Query:\n{prompt}"

# Customize agent behavior
def create_custom_chain(llm):
    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template="Given the context:\n{context}\n\nAnswer the query:\n{query}"
    )
    return LLMChain(llm=llm, prompt=prompt)

def alisen():
    st.markdown(f"""
        ### <span style="color: #ffb31a;">Alisen</span>: the <span style="color: #ffb31a;">A</span>rtificial <span style="color: #ffb31a;">L</span>earning and <span style="color: #ffb31a;">I</span>ntelligence <span style="color: #ffb31a;">S</span>ynthesis <span style="color: #ffb31a;">En</span>gine
    """, unsafe_allow_html=True)

    st.text_area("Heuristix's Alisen - helping you ask better questions and make better decisions",
                 "A virtual advisor, acting as your second-in-charge, helping with data analysis using custom Generative Artificial Cognition and Intelligence using AI Agents.\n\nBespoke to this environment's data and user prompts, and never leaving the environment i.e. trained on specific data and totally secure.")

    st.info("This is decision support: artifically intelligent advanced domain expertise and organizational knowledge. Alisen and her agents are trained on and sourced from your data, so they are experts in your specific environment.")

    # Load and preprocess data
    df = dataingestion.readdata()
    domain_data = df.sample(frac=0.75, random_state=42)  # Use 75% of the dataframe as domain-specific data

    # Fine-tuning and model initialization buttons
    # fine_tuned_model_path = "./fine_tuned_model"
    fine_tuned_model_path = "./local_model"

    if os.path.exists(fine_tuned_model_path):
        st.success("Using pre-fine-tuned model from: ./fine_tuned_model")
    else:
        st.warning("Fine-tuned model not found. You can fine-tune the model using the button below.")
        if st.button("Fine-Tune Model (Warning: Time-Consuming)"):
            with st.spinner("Fine-tuning the model on organizational data..."):
                fine_tuned_model_path = fine_tune_model(domain_data)
                st.success("Fine-tuning completed successfully!")

    # Initialize the local model
    model = LocalLLM(
        api_base="http://localhost:11434/v1",
        model=fine_tuned_model_path
    )

    with st.expander("Sample of data", expanded=False):
        st.write(df.sample(8))

    with st.expander("What can Alisen do?", expanded=False):
        st.image("howdoesalisenwork.png")

    add_vertical_space(3)

    # Setup embedding index
    vector_store = None
    embedding_choice = st.radio("Do you want to use an embedded index?", ("No", "Yes"))

    if embedding_choice == "Yes":
        if not os.path.exists("faiss_index"):
            st.warning("Embedding index is not set up. You can set it up using the button below.")
            if st.button("Setup Embedding Index (Warning: May take time)"):
                with st.spinner("Setting up embedding index..."):
                    documents = domain_data.to_dict(orient="records")  # Convert domain data to records for embedding
                    vector_store = setup_embedding_index(documents)
                    st.success("Embedding index setup completed successfully!")
            else:
                st.stop()  # Stop execution until embedding index is set up
        else:
            vector_store = FAISS.load_local("faiss_index")
            st.success("Using existing embedding index.")

    # Initialize agent
    agent = Agent(df, config={"llm": model})

    st.image('./gui/images/alisenillus.jpg', width=600)

    st.info("Hello! I'm Alisen. My role is to help you understand the data I have stored in my memory, and decide what actions might be best to take based on the insights I can offer.")

    # Prompt input
    prompt = st.text_input("What can I help you with today? Enter a prompt for my analysis here. Be as specific as possible for best results.")

    if st.button("Ask Alisen"):
        if prompt:
            with st.spinner("I'm working on it with my team, after I retrieve the context. Please bear with me for a few moments."):
                try:
                    if vector_store:
                        # Retrieve context
                        retrieved_docs = query_embedding_index(prompt, vector_store)
                        domain_prompt = generate_domain_specific_prompt(prompt, retrieved_docs)
                    else:
                        domain_prompt = prompt

                    # Generate response
                    response = agent.chat(domain_prompt)

                    # Handle graphical outputs
                    if isinstance(plt.gcf(), plt.Figure):
                        st.pyplot(plt.gcf())
                        plt.clf()
                    else:
                        st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
