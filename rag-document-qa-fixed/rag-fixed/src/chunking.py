# src/chunking.py Manish

# Correct import from LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter  # fixed: was langchain.text_splitter

def fixed_chunking(documents):
    """
    Splits documents into chunks of size 800 with 150 overlap.
    Returns a list of chunked document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    return splitter.split_documents(documents)