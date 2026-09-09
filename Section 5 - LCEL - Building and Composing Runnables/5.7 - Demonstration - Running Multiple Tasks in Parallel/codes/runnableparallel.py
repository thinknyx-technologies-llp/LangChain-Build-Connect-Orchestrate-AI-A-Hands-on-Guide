from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableParallel, RunnableLambda

from datetime import datetime

model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite")

parser = StrOutputParser()

def log_start(name, x):
    print(f"{datetime.now().strftime('%H:%M:%S')} -> {name} STARTED")
    return x

def log_end(name, response):
    print(f"{datetime.now().strftime('%H:%M:%S')} -> {name} FINISHED")
    return response

explanation_chain = (
    PromptTemplate.from_template("Explain {topic} in 2 lines.")
    | RunnableLambda(lambda x: log_start("Explantion", x))
    | model
    | parser
    | RunnableLambda(lambda x: log_end("Explantion", x))
)

examples_chain = (
    PromptTemplate.from_template("Give 2 real-world examples for {topic}")
    | RunnableLambda(lambda x: log_start("Examples", x))
    | model
    | parser
    | RunnableLambda(lambda x: log_end("Examples", x))
)

advantages_chain = (
    PromptTemplate.from_template("List 2 advantages of {topic}")
    | RunnableLambda(lambda x: log_start("Advantages", x))
    | model
    | parser
    | RunnableLambda(lambda x: log_end("Advantages", x))
)

parallel_chain = RunnableParallel(
    explanation = explanation_chain,
    example = examples_chain,
    advantages = advantages_chain
)

topic = input("Topic: ")

response = parallel_chain.invoke(
    {
        "topic": topic
    }
)

print(type(response))
# print(response)
print("Explanation", response["explanation"])
print("Examples", response["example"])
print("Advantages", response["advantages"])



