import asyncio

from dotenv import load_dotenv
load_dotenv()

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

async def main():

    client = MultiServerMCPClient(
        {
            "product_server": {
                "command" : "python",
                "transport": "stdio",
                "args": ["mcp_server.py"]
            }
        }
    )

    tools = await client.get_tools()
    print("Available Tools: ")
    for tool in tools:
        print(f"-{tool.name}")

    for tool in tools:
        tool.args_schema.pop("additionalProperties", None)

    model = ChatGoogleGenerativeAI(model= "gemini-3.1-flash-lite")

    system_prompt="""
You are a helpful and conversational AI Product Assistant.

When the user asks to find, search for, or browse
products, use the available product search tool.

Understand the user's request naturally and extract
the relevant product or category from it.

For example:

- "find me laptops" → search for laptops
- "I need headphones" → search for headphones
- "show me some smartphones" → search for smartphones
- "I'm looking for something for gaming"
  → search for relevant products

Always use the product search tool when the user asks
to find or search for products.

When the tool returns multiple products, present all
relevant products to the user.

For each product, include:
- Product name
- Price
- Rating
- Stock availability

Use the conversation history to understand follow-up
questions and references such as:

- "Which one has the highest rating?"
- "Tell me more about the third one."
- "What is its price?"
- "Is it in stock?"
- "What about the second one?"

Never invent product information.

After receiving tool results, answer naturally and
conversationally.

Do not mention MCP, APIs, or internal tool calls.

Keep responses clear, useful, and concise.
"""

    agent = create_agent(
        model = model,
        tools = tools,
        system_prompt = system_prompt
    )

    messages = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        response = await agent.ainvoke(
            {
                "messages": messages
            }
        )

        messages = response["messages"]

        final_response = messages[-1].text
        print("AI: ", final_response)
        print()

if __name__ == "__main__":
    asyncio.run(main())