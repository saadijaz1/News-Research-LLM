import os
import streamlit as st
from dotenv import load_dotenv

# Import functions from model_test_openai.py
from main_openai import (
    initialize_llm,
    load_data,
    split_documents,
    create_embeddings,
    create_vectorstore,
    create_qa_chain,
)

load_dotenv()

# Step 1: Initialize LLM
llm = initialize_llm()

# Step 2: Load Data
file_path = os.path.join(os.path.dirname(__file__), "input_text.txt")
data = load_data(file_path)

# Step 3: Split Documents
docs = split_documents(data)

# Step 4: Create Embeddings
embeddings = create_embeddings(docs)

# Step 5: Create Vector Store
vectorstore_google = create_vectorstore(docs, embeddings)

# Step 6: Create QA Chain
chain = create_qa_chain(llm, vectorstore_google)

# Step 7: Streamlit App
st.title("Interactive QA Dashboard")

# Streamlit Sidebar
st.sidebar.header("User Input")

# User input for question
user_query = st.sidebar.text_input("Enter your question:")

# Interactive Query
if st.sidebar.button("Ask Question"):
    try:
        result = chain(user_query)
        st.success("Answer: {}".format(result["result"]))
    except IndexError as e:
        st.error(
            "Error: Unable to retrieve an answer for the given question. Exception: {}".format(
                e
            )
        )
        try:
            st.info("Model output: {}".format(result["result"]))
        except AttributeError:
            st.info("Model output does not contain 'result' attribute.")
