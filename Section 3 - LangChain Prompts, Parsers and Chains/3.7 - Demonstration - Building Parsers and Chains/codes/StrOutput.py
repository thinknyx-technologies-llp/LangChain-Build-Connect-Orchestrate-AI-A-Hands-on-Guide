from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

prompt = PromptTemplate.from_template(
    "Explain {topic} in 3 to 5 bullet points."
)

final_prompt = prompt.invoke(
    {
        "topic": "Neural Networks"
    }
)

response = model.invoke(final_prompt)
print(response.content[0]["text"])

parser = StrOutputParser()
final_response = parser.invoke(response)
print(final_response)