import os
import streamlit as st
import concurrent
from dotenv import load_dotenv

# Import functions from model_test_openai.py
from main_palm import (
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

# Input field for the user to enter a question
user_query = st.text_input("Enter your question:")

# List to store chat history
chat_history = []

if user_query:
    try:
        # Run the question through the QA chain
        result = chain(user_query)

        # Append the user query and response to chat history
        chat_history.append({"user": user_query, "response": result["result"]})

    except Exception as e:
        # Handle errors
        st.error(f"Error: {e}")

# Display the entire chat history
for entry in chat_history:
    st.text(f"User: {entry['user']}")
    st.text(f"Response: {entry['response']}")
