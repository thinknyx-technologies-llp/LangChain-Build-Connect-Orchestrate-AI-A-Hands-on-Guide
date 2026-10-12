from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from langchain_core.runnables import RunnableParallel

model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite")

parser = JsonOutputParser()

code = """
def is_palindrome(s):
    return s == s[::-1]
"""

review_prompt = PromptTemplate.from_template(

"""
You are a senior Code Reviewer.
Review the following code and provide:
- Code explanation.
- Bugs (if any).
- Time Complexity.
- Space Complexity.
- Suggestions.

Code:
{code}

"""    
)

explain_prompt = PromptTemplate.from_template(
    """
Explain this code in simple language.
Code: 
{code}
"""
)

optimize_prompt = PromptTemplate.from_template(
    """
Optimize the following code.

Return: 
- Optimized Code.
- Time Complexity.
- Space Complexity.
- Reason for optimization.

Code:
{code}
"""
)

pipeline = RunnableParallel(
    Review = review_prompt | model | parser,
    Explantion = explain_prompt | model | parser,
    Optimization = optimize_prompt | model | parser
)

result = pipeline.invoke({
    "codes" : code
})

for title, content in result.items():
    print("Title: ", title.upper())
    print("Content: ".upper())
    print(content)
    print()
