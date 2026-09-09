from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import re

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

model = ChatGoogleGenerativeAI(model  = "gemini-3.1-flash-lite")

def clean_text(text: str) -> str:

    text = text.strip()
    text = re.sub(r"\?+", "?", text)
    text = re.sub(r"\s+", " ", text)

    if text:
        text = text[0].upper() + text[1: ]

    return text

# user_input = input("Question: ")

# res = clean_text(user_input)
# print(res)

cleaner = RunnableLambda(clean_text)

prompt = PromptTemplate.from_template(
    """
Answer the following question in 3 lines.

Question: {question}
"""
)

chain = (
    cleaner
    | {"question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

question = input("Question: ")
response = chain.invoke(question)

print(response)

