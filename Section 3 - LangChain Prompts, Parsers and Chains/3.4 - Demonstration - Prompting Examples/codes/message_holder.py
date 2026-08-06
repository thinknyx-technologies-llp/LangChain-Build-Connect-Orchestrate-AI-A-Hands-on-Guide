from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant for tuting python"),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ]
)

messages = prompt.invoke(
    {
        "history":[
            ("human", "What is Python?"),
            ("ai", "Python is a programming Language.")
        
        ],
        "question":"What is a dictionary?"
    }
)

print(messages)


response = model.invoke(messages)
print(response.content[0]["text"])