### chat_model_2.py

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

model = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation",
    huggingfacehub_api_token = API_KEY,
)

cm = ChatHuggingFace(llm = model,)
response = cm.invoke("Who is the Prime Minister of India?")

print(response.content)
print(type(response))


### chat_model3.py

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os 
load_dotenv()

from pathlib import Path 
import os  

project_root = Path(__file__).parent  
os.environ["HF_HOME"] = str(project_root / "huggingface") 

API_KEY = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation"
)

cm = ChatHuggingFace(llm = llm)

response = cm.invoke("What is the capital of France?")

print(response.content)