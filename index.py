from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
import os
from dotenv import load_dotenv

load_dotenv()


api_key = os.getenv("GOOGLE_API_KEY")


pdf_path = Path(__file__).parent/ "nodejs.pdf"

#Loading the PDF document
loader = PyPDFLoader(pdf_path)
docs = loader.load()

#Splitting the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_documents(documents=docs)

#Embedding the chunks using Gemini Embeddings and storing them in Qdrant vector database

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en"
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="my_documents",
)

print("Document loaded, split into chunks, embedded and stored in Qdrant vector database successfully!")
