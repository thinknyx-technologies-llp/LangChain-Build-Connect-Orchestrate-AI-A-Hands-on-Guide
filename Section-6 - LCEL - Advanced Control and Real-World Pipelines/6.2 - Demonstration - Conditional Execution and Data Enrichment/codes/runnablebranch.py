from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite")

prompt = PromptTemplate.from_template(
    """
Question Type: {category}

Answer the following question according to the defined type.

Question : {question}
"""
)

def get_def(x):
    print("Routing to DEFINITION....")
    return "Definition"

def get_comparison(x):
    print("Rounting to COMPARISON...")
    return "Comparison"

def get_coding(x):
    print("Rounting to CODING...")
    return "Coding"

def get_gen(x):
    print("Rounting to GENERAL...")
    return "General"


branch = RunnableBranch(
    (
        lambda x: "what is" in x["question"].lower(),
        RunnableLambda(get_def)
    ),
    (
        lambda x: "difference" in x["question"].lower(),
        RunnableLambda(get_comparison)
    ),
    (
        lambda x: "code" in x["question"].lower(),
        RunnableLambda(get_coding)
    ),

    RunnableLambda(get_gen)
)


assign = RunnablePassthrough.assign(
    category = branch
)


chain = (
    assign
    | prompt
    | model
    | StrOutputParser()
)

user_input = input("Input: ")

question = {
    "question" : user_input
}

response = chain.invoke(question)

print(response)