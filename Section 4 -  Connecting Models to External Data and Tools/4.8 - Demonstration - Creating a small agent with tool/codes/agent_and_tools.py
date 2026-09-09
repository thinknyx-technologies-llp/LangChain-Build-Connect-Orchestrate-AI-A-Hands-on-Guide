from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent

model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite")

@tool
def get_weather(city: str) -> str:
    """
    Returns the weather information for a city.
    """

    weather = {
        "delhi" : "Sunny, 36 degrees C",
        "mumbai": "Cloudy, 30 degrees C",
        "chennai" : "Rainy, 29 degrees C",
        "bangalore" : "Pleasant, 25 degrees C",
        "kolkata" : "Humid, 32 degrees C"
    }

    return weather.get(
        city.lower(), 
        "Weather information is not available for this city."
    )


agent = create_agent(
    model = model,
    tools = [get_weather]
)

question = input("Question: ")
response = agent.invoke(
    {
        "messages" : [
            {
                "role": "user",
                "content": question
            }
        ]
    }
)

print(type(response))
print("\n")
# print(response)
print(response["messages"][-1].content[0]["text"])