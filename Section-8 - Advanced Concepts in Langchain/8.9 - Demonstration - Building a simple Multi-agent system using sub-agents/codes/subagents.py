from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

research_agent = create_agent(
    model=model,
    system_prompt="""
    You are  a Research Specialist.

    Your job is to handle:
    - General Knowledge questions.
    - Concept explanations.
    - Technical explanations.
    - Research-oriented Questions.
    
    Give clear and concise answers.
    Return only the final answer.
"""
)

math_agent = create_agent(
    model=model,
    system_prompt="""
You are a Math Specialist.

Your job is to handle:
- Mathematical Calculations.
- Arithmetic
- Mathematical Problem Solving.

Show the calculation clearly and provide the final answer.

Return on the answer.
"""
)

@tool
def research_subagent(query: str) -> str:
    """
    Delegate research, explanation, definitions, and general knowledge question to the Research Agent.
    """
    print("\nResearch Subagent is working...")

    result = research_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        }
    )

    return result["messages"][-1].text

@tool
def math_subagent(query: str) -> str:
    """
    Delegate mathematical calculations and mathematical problem solving to the Math Agent.
    """
    print("\nMath Subagent is working...")

    result = math_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        }
    )

    return result["messages"][-1].text

main_agent= create_agent(
    model=model,
    tools=[
        research_subagent,
        math_subagent
    ],
    system_prompt="""
    You are the Main Agent.

    You have access to two specialized subagents:

    1. research_subagent
       Handles:
       - Research
       - General knowledge
       - Concept explanations
       - Technical explanations

    2. math_subagent
       Handles:
       - Mathematical calculations
       - Arithmetic
       - Mathematical problem solving

    Decide which subagent is appropriate for the
    user's request.

    Delegate the task to the appropriate subagent
    and use its result to provide the final answer.

    Do not perform the specialized task yourself
    when an appropriate subagent is available.
"""
)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break

    try:
        result= main_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            }
        )

        print("\nMain Agent: ", result["messages"][-1].text)

    except Exception as e:
        print("Error: ", e)

