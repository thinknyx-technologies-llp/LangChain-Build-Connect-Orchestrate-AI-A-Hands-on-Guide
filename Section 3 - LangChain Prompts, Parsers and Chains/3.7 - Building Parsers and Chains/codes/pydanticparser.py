from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite")

class Person(BaseModel):
    name: str = Field(description = "Person's Name")
    age: int = Field(description = "Person's Age")
    profession: str = Field(description = "Person's Profession")

parser = PydanticOutputParser(pydantic_object=Person)

prompt = PromptTemplate.from_template(
    """
Extract the following information from the text.
{format_instructions}

Text: {text}
"""
)

chain = prompt | model | parser

response = chain.invoke({
    "text" : "Elon Musk is a 54 year old entrepreneur",
    "format_instructions": parser.get_format_instructions()
})

print(parser.get_format_instructions())
print("\n")
print(response)
print(type(response))