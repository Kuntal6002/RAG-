from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333", collection_name="my_documents", embedding=embeddings
)

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

user_query = input("Ask Something: ")

search_result = vector_db.similarity_search(query=user_query)

context = "\n".join(
    [
        f"Page Number: {result.metadata['page_label']}, Page Content: {result.page_content},\nFile Location: {result.metadata['source']}"
        for result in search_result
    ]
)

SYSTEM_PROMPT = f"""You are a helpful assistant who answer user query based on the the available text context retrieved from a PDF file along with its page_contents and number. You should only answer the user based on the following context and navigate the user to the right page number to know more about the answer. If you don't know the answer based on the following context then say you don't know and do not try to make up an answer.{context}"""

response = client.chat.completions.create(
    model="gemini-3-flash-preview",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ],
)

print(response.choices[0].message.content)
