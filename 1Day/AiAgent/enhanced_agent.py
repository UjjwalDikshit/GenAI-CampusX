from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community. tools import DuckDuckGoSearchRun
import requests

search_tool = DuckDuckGoSearchRun()

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

@tool
def get_weather_data(city: str) -> str:
    """This function fetches tha cuenant wasthar data for a given city"""
    url = f'https://api.weatherstack.com/current?access_key=4d1d8ae207a8c845a52df8a67bf3623e&query={city}'

    response = requests.get(url)

    return response.json()

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt


# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)


# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True
)

# Step 5: Invoke
response = agent_executor.invoke({"input": "Find the capital of Madhya Pradesh, then find it's current weather condition"})
print(response)

response['output']