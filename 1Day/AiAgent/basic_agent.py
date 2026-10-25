# deactivate
# python -m venv venv_classic
# venv_classic\Scripts\activate
# python -m pip install --upgrade pip

# python -m pip install "langchain==0.3.27" "langchain-community==0.3.27" "langchain-google-genai==2.1.12" "duckduckgo-search" "langchainhub" "python-dotenv"




# ============================================================
# PROJECT: Internet Browsing Agent using LangChain + Gemini
# ============================================================
# PURPOSE:
# Create an AI agent that can search the internet using
# DuckDuckGo and answer questions using the search results.
#
# Example:
# Question: "3 ways to reach Goa from Delhi"
# Agent: Searches the internet -> reads results -> generates
#       a final answer using the Gemini model.
#
# WORKFLOW:
# User Question -> ReAct Agent (Gemini) -> Search Tool -> Search Results
#               -> Agent processes results -> Final Answer
# ============================================================


# STEP 1: IMPORT REQUIRED LIBRARIES

from dotenv import load_dotenv
# PURPOSE:
# Loads environment variables from a .env file.
# We can store API keys and configuration values in .env
# instead of hardcoding them directly into our Python code.
# Example: GOOGLE_API_KEY="your_api_key"


from langchain_google_genai import ChatGoogleGenerativeAI
# PURPOSE:
# Connects LangChain with Google's Gemini chat models.
# Gemini acts as the BRAIN of our agent.
#
# WHAT DOES THE MODEL DO?
# 1. Understands the user's question.
# 2. Decides whether a tool is needed.
# 3. Chooses which tool to use and generates its input.
# 4. Processes the tool's results.
# 5. Generates the final answer.
#
# IMPORTANT:
# The model itself does not automatically browse the internet.
# We provide a search tool so the agent can retrieve information.


from langchain_community.tools import DuckDuckGoSearchRun
# PURPOSE:
# Provides a search tool that searches the internet through
# DuckDuckGo and returns text containing search results.
#
# WHY DO WE NEED IT?
# Gemini's built-in knowledge may not contain the latest
# information. A search tool helps retrieve potentially
# current information from the web.
#
# NOTE:
# Search results can be incomplete or inaccurate, so the
# agent should interpret them carefully.


from langchain.agents import create_agent ,AgentExecutor

# PURPOSE:
# These are the main components used to create and run
# our classic ReAct agent.
#
# create_react_agent:
# Connects the language model, tools, and prompt into an agent.
#
# AgentExecutor:
# Runs the agent's tool-use loop. It passes observations back
# to the agent and continues until a final answer is produced
# or an execution limit is reached.


from langchain import hub
# PURPOSE:
# Allows us to retrieve reusable prompts from LangChain Hub.
#
# A prompt defines the instructions and format that guide
# the model's behavior.
#
# The ReAct prompt teaches the model how to structure its
# tool-use decisions and interpret tool observations.


# STEP 2: LOAD ENVIRONMENT VARIABLES

load_dotenv()

# WHY?
# Loads variables defined in the .env file into the process
# environment so integrations can access their configuration.
#
# Make sure your Google API key is configured correctly.
# Never share your API key publicly.


# STEP 3: INITIALIZE THE LANGUAGE MODEL (LLM)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# WHAT IS AN LLM?
# LLM stands for Large Language Model.
# It processes natural language and generates text.
#
# WHY GEMINI 2.5 FLASH?
# It is a Gemini model designed for fast responses.
# We use it as the reasoning and language-generation
# component of this agent.
#
# IMPORTANT DISTINCTION:
# The LLM is the BRAIN.
# The search tool provides ACCESS TO SEARCH RESULTS.
# The agent connects the two.


# STEP 4: CREATE THE INTERNET SEARCH TOOL

search_tool = DuckDuckGoSearchRun()

# WHAT IS A TOOL IN LANGCHAIN?
# A tool is a function or capability that an agent can invoke
# to perform a task outside ordinary text generation.
#
# Here, DuckDuckGoSearchRun searches for information online.
#
# Example:
# search_tool.invoke("top news in India today")
#
# The tool returns text containing search results.
# The agent can then use those results to formulate an answer.


# OPTIONAL: TEST THE SEARCH TOOL DIRECTLY

# results = search_tool.invoke("top news in India today")
# print(results)

# WHY TEST IT SEPARATELY?
# It helps us verify that the search tool works before
# integrating it into the agent.
#
# If this fails, troubleshoot the search dependency or
# network access before debugging the agent itself.


# STEP 5: PULL THE REACT PROMPT FROM LANGCHAIN HUB

prompt = hub.pull("hwcase17/react")

# WHAT IS A PROMPT?
# A prompt is a set of instructions provided to the model.
# It helps guide how the model should respond and use tools.
#
# WHAT IS REACT?
# ReAct means REASON + ACT.
#
# The agent follows a loop:
#
# 1. THINK/REASON:
#    Interpret the user's question and determine the next step.
#
# 2. ACTION:
#    Select a tool, such as DuckDuckGoSearchRun.
#
# 3. OBSERVATION:
#    Receive the result returned by the tool.
#
# 4. REASON AGAIN:
#    Decide whether more tool use is needed or the results
#    are sufficient to answer the question.
#
# 5. FINAL ANSWER:
#    Return a user-facing response.
#
# The prompt defines the format used by the classic ReAct
# agent, including the tool names and intermediate scratchpad.
#
# IMPORTANT:
# The prompt ID must exist on LangChain Hub, and the prompt
# must be compatible with create_react_agent.


# STEP 6: CREATE THE REACT AGENT

agent = create_react_agent(
    llm=model,
    tools=[search_tool],
    prompt=prompt
)

# WHAT HAPPENS HERE?
#
# llm=model:
# Provides Gemini as the language model that selects actions
# and generates responses.
#
# tools=[search_tool]:
# Makes the DuckDuckGo search tool available to the agent.
# The agent can only use tools that are supplied to it.
#
# prompt=prompt:
# Supplies the instructions that guide the agent's behavior.
#
# IMPORTANT:
# This step CREATES the agent; it does not execute the
# complete user request yet.
#
# The agent's tool-use behavior depends on the prompt,
# model output, and the tools provided.


# STEP 7: CREATE THE AGENT EXECUTOR

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True
)

# WHAT IS AN AGENT EXECUTOR?
# The agent determines the next action.
# The executor runs that action and returns its observation
# to the agent so it can continue.
#
# WHY PROVIDE THE TOOL AGAIN?
# The agent uses the tool definitions to decide which action
# to request. The executor needs the actual tools to execute
# those requests.
#
# WHY verbose=True?
# Displays intermediate execution information, such as
# tool actions and observations, which is useful for learning
# and debugging the agent workflow.
#
# NOTE:
# Verbose output is useful for development. Avoid exposing
# sensitive information in logs.


# STEP 8: SEND A QUESTION TO THE AGENT

response = agent_executor.invoke(
    {"input": "3 ways to reach Goa from Delhi"}
)

# WHAT DOES invoke() DO?
# It sends the input to the executor and starts the agent loop.
#
# The input is a dictionary because the classic ReAct prompt
# generally expects a variable named "input".
#
# POSSIBLE EXECUTION:
#
# User asks about ways to reach Goa from Delhi
#            |
#            v
# Gemini interprets the question
#            |
#            v
# Agent decides to search the web
#            |
#            v
# DuckDuckGo returns search results
#            |
#            v
# Gemini processes the observations
#            |
#            v
# Agent generates the final answer
#
# The agent may use the search tool more than once if needed.
# Its exact behavior depends on the prompt and model output.


# STEP 9: PRINT ONLY THE FINAL ANSWER

print(response["output"])

# WHY response["output"]?
# invoke() generally returns a dictionary containing the
# final answer and possibly other execution information.
#
# response["output"] extracts the final answer as a string.
#
# If you instead use print(response), you will generally see
# the entire returned dictionary.
#
# FINAL RESULT:
# A text answer describing three possible ways to travel
# from Delhi to Goa, based on the information retrieved.
