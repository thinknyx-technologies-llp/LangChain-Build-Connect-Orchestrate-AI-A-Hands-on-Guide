from typing import Any
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, before_agent, after_agent
from langchain.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.runtime import Runtime

@before_agent
def input_guardrail(
    state: AgentState,
    runtime: Runtime
) -> dict[str, Any] | None:

    user_message  =state["messages"][-1].text.lower()

    blocked_words = [
        "hack", "malware", "exploit", "password"
    ]

    for word in blocked_words:
        if word in user_message:
            print("INPUT Guardrail: Request Blocked!")
            return{
                "message": [
                    AIMessage("I can't help you with the request involing words like  hack, malware, password and exploit")
                ],
                "jump_to": "end"
            }
    return None

@after_agent
def output_guardrail(
    state: AgentState,
    runtime: Runtime
) -> dict[str, Any] | None:

    last_message = state["messages"][-1]

    response = last_message.text.lower()

    blocked_words = [
        "password", "credit card", "secret key"
    ]

    for word in blocked_words:
        if word in response:
            print("OUTPUT GUARDRAIL: Unsafe response detected!")
            return {
                "messages": [
                    AIMessage("Im sorry, but I can't provide that information")
                ]
            }

    return None

model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite")

agent = create_agent(
    model = model,
    middleware=[
        input_guardrail,
        output_guardrail
    ]
)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    try:
        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
         }
        )

        print("Agent: ", result["messages"][-1].text)

    except Exception as e:
        print("Error:", e)