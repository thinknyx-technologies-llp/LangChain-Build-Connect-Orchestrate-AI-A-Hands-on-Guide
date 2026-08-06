from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
prompt = PromptTemplate.from_template(
    "Explain {topic} in simple words"
)

final_prompt = prompt.invoke({
    "topic": "Neural Networks"
})
print(final_prompt.text)

response = model.invoke(final_prompt)

print(response.content[0]["text"])