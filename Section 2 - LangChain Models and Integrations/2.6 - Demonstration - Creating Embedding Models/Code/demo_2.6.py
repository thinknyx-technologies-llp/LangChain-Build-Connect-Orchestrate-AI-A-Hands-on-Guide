##### embedding_model.py

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")
vector = embeddings.embed_query("What is Artificial Intelligence?")

print(len(vector))
print(vector[:10])


##### embedding_model_2.py

from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vector = embeddings.embed_query("What is Artificial Intelligence?")

print(len(vector))
print(vector[:10])