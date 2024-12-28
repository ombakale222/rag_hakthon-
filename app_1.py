import os
import asyncio
import nest_asyncio
import requests
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import fitz  # pymupdf for PDF processing
import streamlit as st  # Streamlit for the UI

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

nest_asyncio.apply()

# Initialize Hugging Face model and tokenizer
model_name = "distilbert-base-uncased"
# You can choose other models from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    url = "https://api.hyperbolic.xyz/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJvbWJha2FsZTIyQGdtYWlsLmNvbSIsImlhdCI6MTczMjY0MDMxMH0.CCgwKM-3WagOVXXlcBMsf-IzUWrOmFz3ZTz9SLVT0fo"  # Replace with your actual token
    }
    data = {
        "messages": [
            {
                "role": "system",
                "content": "You are an intelligent assistant specializing in PDF content retrieval. Your primary role is to efficiently extract, search, and summarize content from PDF documents based on user queries."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "max_tokens": 131072,
        "temperature": 0.1,
        "top_p": 1
    }

    try:
        response = requests.post(url, headers=headers, json=data)

        # Check for successful response
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response text: {response.text}")
            return ""

        # Log the raw response to debug
        print("Response received from API:")
        print(response.text)

        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")

    except requests.exceptions.RequestException as e:
        # Catch network or connection-related errors
        print(f"Request exception occurred: {e}")
        return ""
    except ValueError as e:
        # Catch JSON decoding errors
        print(f"JSON decoding error occurred: {e}")
        return ""
    except Exception as e:
        # Catch any other unforeseen errors
        print(f"An error occurred: {e}")
        return ""

async def embedding_func(texts: list[str]) -> np.ndarray:
    """
    Generates embeddings for a list of texts using a Hugging Face model.

    Args:
        texts (list[str]): List of input text strings.

    Returns:
        np.ndarray: Array of embeddings.
    """
    # Tokenize the input texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

    with torch.no_grad():
        # Get the embeddings from the model
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Average pool the token embeddings

    # Convert embeddings to numpy array
    return embeddings.numpy()

async def get_embedding_dim():
    test_text = text
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using pymupdf (fitz)."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()  # Extract text from each page
    return text

def main():
    st.title("PDF Query Assistant")

    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

    if uploaded_file is not None:
        # Save the uploaded file
        pdf_path = os.path.join(WORKING_DIR, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success("PDF uploaded successfully!")

        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_path)

        query = st.text_input("Enter your query")

        if st.button("Get Answer"):
            try:
                # Initialize RAG
                embedding_dimension = 768
                rag = LightRAG(
                    working_dir=WORKING_DIR,
                    llm_model_func=llm_model_func,
                    embedding_func=EmbeddingFunc(
                        embedding_dim=embedding_dimension,
                        max_token_size=8192,
                        func=embedding_func,
                    ),
                )

                asyncio.run(rag.ainsert(pdf_text))

                # Query the PDF
                answer = asyncio.run(rag.aquery(query, param=QueryParam(mode="naive")))

                st.write("### Answer:")
                st.write(answer)

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
