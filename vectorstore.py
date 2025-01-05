from qdrant_client import qdrant_client
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv, dotenv_values
import os

load_dotenv()

def create_retriever(doc_list, embed_model):
    qdrant = QdrantVectorStore.from_documents(
    doc_list,
    embed_model,
    url = os.getenv("qdrant_url"),
    api_key = os.getenv("qdrant_key"),
    collection_name = "Steve_McConnel_code_book"
    )

    retriever = qdrant.as_retriever()
    return retriever
