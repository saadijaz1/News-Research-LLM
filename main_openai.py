import os
import streamlit as st

from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders.text import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()


def initialize_llm():
    return OpenAI(temperature=0.9, max_tokens=500)


def load_data(file_path):
    file_path = os.path.join(os.path.dirname(__file__), "input_text.txt")
    loader = TextLoader(file_path=file_path, encoding="utf-8")
    data = loader.load()
    return data


def split_documents(data):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","], chunk_size=512, chunk_overlap=125
    )
    docs = text_splitter.split_documents(data)
    return docs


def create_embeddings(docs):
    return OpenAIEmbeddings()


def create_vectorstore(docs, embeddings):
    return FAISS.from_documents(docs, embeddings)


def create_qa_chain(llm, retriever, input_key="query", return_source_documents=False):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever.as_retriever(),
        input_key=input_key,
        return_source_documents=return_source_documents,
    )


def interactive_query(chain):
    while True:
        user_query = input("Enter your question (or 'exit' to quit): ")

        if user_query.lower() == "exit":
            break

        try:
            result = chain(user_query)
            print(result["result"])
        except IndexError as e:
            print(
                f"Error: Unable to retrieve an answer for the given question. Exception: {e}"
            )

            try:
                print("Model output:", result["result"])
            except AttributeError:
                print("Model output does not contain 'result' attribute.")


if __name__ == "__main__":
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

    # Step 7: Interactive Query
    interactive_query(chain)
