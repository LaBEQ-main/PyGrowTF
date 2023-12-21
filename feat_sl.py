import streamlit as st
import os
import pickle
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
# Set OpenAI API Key
os.environ['OPENAI_API_KEY'] = "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"

# Initialize LLM with required params
llm = OpenAI(temperature=0.5, max_tokens=500)

# Streamlit app
def main():
    st.title("PyGrowTF materialS Search Tool")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        # Read and process the uploaded file
        data = process_document(uploaded_file)

        # Search query input
        query = st.text_input("Enter your search query")

        if query:
            # Perform search and get results
            results = perform_search(query, data)
            st.write(results)
class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}

def get_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)

    # Extract text from each page
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text

def process_document(uploaded_file):
    # Read the uploaded file
    #converted_file=get_pdf_text(uploaded_file)
    #text = converted_file.read().decode('latin-1')
    text=get_pdf_text(uploaded_file)
    # Splitting the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # Creating Document instances with metadata
    documents = [Document(chunk, {"source": "UploadedFile"}) for chunk in chunks]

    # Create embeddings for the chunks
    embeddings = OpenAIEmbeddings()
    vectorindex_openai = FAISS.from_documents(documents, embeddings)

    # Save and load the vector index
    file_path = "vector_index.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(vectorindex_openai, f)

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorIndex = pickle.load(f)

    return vectorIndex


def perform_search(query, vectorIndex):
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorIndex.as_retriever())

    langchain.debug = True
    result = chain({"question": query}, return_only_outputs=True)

    return result

if __name__ == "__main__":
    main()
