from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from src.retriever import get_retriever
from src.generator import build_qa_chain

def load_pipeline():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        "vector_store", 
        embeddings,
        allow_dangerous_deserialization=True  # ← required in langchain-community >= 0.0.27
    )
    retriever = get_retriever(vectorstore)
    return build_qa_chain(retriever)