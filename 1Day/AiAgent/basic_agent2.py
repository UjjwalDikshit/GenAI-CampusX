
# ============================================================
# PROJECT: Internet Browsing Agent using LangChain + Gemini
# VERSION B: CLASSIC LANGCHAIN API (0.3.x)
# ============================================================
# PURPOSE:
# Create a classic ReAct agent that searches the internet
# and uses Gemini to answer questions.
#
# WORKFLOW:
# User Question -> ReAct Agent -> Search Tool
#               -> Observation -> Agent reasons again
#               -> Final Answer
# ============================================================


# STEP 1: IMPORT REQUIRED LIBRARIES

from dotenv import load_dotenv
# Loads API keys and configuration from the .env file.


from langchain_google_genai import ChatGoogleGenerativeAI
# Connects LangChain with Gemini.
# Gemini interprets questions and selects actions.


from langchain_community.tools import DuckDuckGoSearchRun
# Provides the classic DuckDuckGo search integration.
# This belongs to langchain-community, which is deprecated
# in favor of standalone integrations for newer development.
# We retain it here to learn the classic API.


from langchain.agents import create_react_agent, AgentExecutor
# create_react_agent:
# Connects the LLM, tools, and ReAct prompt.
#
# AgentExecutor:
# Executes tool calls and passes observations back to
# the agent until a final answer is generated.


from langchain import hub
# Retrieves reusable prompts from LangChain Hub.
# We use a ReAct prompt that defines the expected
# reasoning-and-action format.


# STEP 2: LOAD ENVIRONMENT VARIABLES

load_dotenv()

# WHY?
# Loads GOOGLE_API_KEY from the .env file.
# Keep your real API key private.


# STEP 3: INITIALIZE THE LANGUAGE MODEL

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# Gemini is the BRAIN of the agent.
# It interprets the question, selects a tool, and uses
# the tool's results to generate an answer.


# STEP 4: CREATE THE INTERNET SEARCH TOOL

search_tool = DuckDuckGoSearchRun()

# A tool gives the agent an external capability.
# This tool searches DuckDuckGo and returns text results.
#
# OPTIONAL: TEST THE SEARCH TOOL
# results = search_tool.invoke("top news in India today")
# print(results)


# STEP 5: PULL THE REACT PROMPT FROM LANGCHAIN HUB

prompt = hub.pull("hwcase17/react")

# ReAct means REASON + ACT.
#
# Conceptual loop:
# 1. Agent interprets the question.
# 2. Agent selects a tool and provides its input.
# 3. Executor runs the tool.
# 4. Tool output becomes an observation.
# 5. Agent decides whether to continue or answer.
#
# The prompt must be compatible with create_react_agent.
# Classic ReAct prompts normally require:
# tools, tool_names, and agent_scratchpad.


# STEP 6: CREATE THE REACT AGENT

agent = create_react_agent(
    llm=model,
    tools=[search_tool],
    prompt=prompt
)

# llm=model:
# Supplies Gemini as the language model.
#
# tools=[search_tool]:
# Makes the search tool available to the agent.
#
# prompt=prompt:
# Supplies the instructions and formatting for ReAct.
#
# This creates the agent but does not run the question yet.


# STEP 7: CREATE THE AGENT EXECUTOR

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True,
    max_iterations=5
)

# AgentExecutor runs the agent's tool-use loop.
#
# Why provide the tool again?
# The agent chooses an action, while the executor needs
# the actual tool implementation to run that action.
#
# verbose=True:
# Displays intermediate actions and observations.
#
# max_iterations=5:
# Limits the number of agent iterations to avoid
# an excessively long execution loop.


# STEP 8: SEND THE QUESTION TO THE AGENT

response = agent_executor.invoke(
    {"input": "3 ways to reach Goa from Delhi"}
)

# The classic API typically accepts an "input" dictionary.
# The executor runs the agent and manages tool execution.
#
# The agent may search for relevant travel options,
# receive observations, and generate its final answer.


# STEP 9: PRINT THE FINAL ANSWER

print(response["output"])

# The classic AgentExecutor normally returns a dictionary
# containing an "output" key with the final answer.
