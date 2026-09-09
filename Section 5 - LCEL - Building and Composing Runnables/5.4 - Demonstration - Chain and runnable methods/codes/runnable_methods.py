from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import asyncio

model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite")
prompt = PromptTemplate.from_template(
    "Explain {topic} briefly in 2 lines"
)

parser = StrOutputParser()

chain = prompt | model | parser

print("Type of chain: ", type(chain))
response = chain.invoke(prompt.invoke({
    "topic": "Electric Vehicles"
}))
print(response)

# batch method
topics = [
    {"topic": "Machine learning"},
    {"topic" : "Neural Networks"},
    {"topic": "Transformers"}
]
responses = chain.batch(topics)
for i, response in enumerate(responses, start=1):
    print(f"\n Response: {i}")
    print(response)

# stream method

for chunk in chain.stream(prompt.invoke({
    "topic": "Cloud Computing"
})):
    print(chunk, end="", flush = True)

# ainvoke method - asynchronous invoke

async def main():
    response = await chain.ainvoke(prompt.invoke({
        "topic" : "Generative AI"
    }))
    print(response)

asyncio.run(main())