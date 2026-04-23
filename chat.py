from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="my_documents",
    embedding=embeddings
)
