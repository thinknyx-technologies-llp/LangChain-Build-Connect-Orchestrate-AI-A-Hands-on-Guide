from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
response = model.invoke("Explain LangChain in Simple Words.")

print(response.content[0]["text"])