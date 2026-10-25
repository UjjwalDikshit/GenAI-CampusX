
import os
import requests
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

load_dotenv()

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",  # Replace with a supported model if needed
    temperature=0
)

# Initialize search tool
search_tool = DuckDuckGoSearchRun()


# Define weather tool
@tool
def get_weather_data(city: str) -> str:
    """Fetch current weather data for a given city."""

    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return "Weather API key is missing. Set WEATHERSTACK_API_KEY in your .env file."

    url = "https://api.weatherstack.com/current"

    try:
        response = requests.get(
            url,
            params={
                "access_key": api_key,
                "query": city
            },
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return f"Weather API error: {data['error']}"

        current = data.get("current", {})
        location = data.get("location", {})

        return (
            f"City: {location.get('name', city)}\n"
            f"Country: {location.get('country', 'Unknown')}\n"
            f"Temperature: {current.get('temperature', 'Unknown')} °C\n"
            f"Condition: {current.get('weather_descriptions', ['Unknown'])[0]}\n"
            f"Humidity: {current.get('humidity', 'Unknown')}%\n"
            f"Wind speed: {current.get('wind_speed', 'Unknown')} km/h"
        )

    except requests.RequestException as e:
        return f"Unable to fetch weather data: {e}"
    except (ValueError, KeyError, IndexError, TypeError) as e:
        return f"Unable to process weather response: {e}"


# Create modern LangChain agent
agent = create_agent(
    model=model,
    tools=[search_tool, get_weather_data],
    system_prompt=(
        "You are a helpful assistant. "
        "Use the search tool to identify the capital of the requested state. "
        "Then use the weather tool to fetch the current weather for that city. "
        "Clearly distinguish live weather data from general information."
    )
)

# Invoke the agent
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Find the capital of Madhya Pradesh, "
                    "then find its current weather condition."
                )
            }
        ]
    }
)

# Print the final answer
print(response["messages"][-1].content)
