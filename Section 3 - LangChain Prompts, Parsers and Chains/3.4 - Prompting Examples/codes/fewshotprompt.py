from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# examples = [
#     {
#         "review" : "This movie was fantastic.",
#         "sentiment" : "positive"
#     },
#     {
#         "review": "The acting was disappointing.",
#         "sentiment" : "negative"
#     }
# ]

# prompt = PromptTemplate.from_template(
#     "review: {review}\n Sentiment: {sentiment}"
# )

# few_shot_prompt = FewShotPromptTemplate(
#     examples = examples,
#     example_prompt = prompt,
#     suffix = "Review: {review}\n Sentiment:",
#     input_variables = ["review"]
# )

# final_prompt = few_shot_prompt.invoke(
#     {
#         "review" : "The visuals were good."
#     }
# )

# print(type(final_prompt))
# print(final_prompt)
# print()
# print("PromptValue: ", final_prompt.text)


# response = model.invoke(final_prompt)
# print(response.content[0]["text"])

prompt = PromptTemplate.from_template(
    """
Explain 2 lines about topic: {topic}
"""
)
prompt_value = prompt.invoke({"topic": "Artificial Intelligence"}) 

print(type(prompt_value))
print(prompt_value)

print(prompt_value.text)
