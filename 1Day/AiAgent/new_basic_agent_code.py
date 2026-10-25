
# ============================================================
# PROJECT: Internet Browsing Agent using LangChain + Gemini
# VERSION A: NEW LANGCHAIN API (1.x)
# ============================================================
# PURPOSE:
# Search the internet using DuckDuckGo and answer questions
# using Gemini 2.5 Flash.
#
# WORKFLOW:
# User Question -> Gemini Agent -> Search Tool -> Results
#               -> Gemini processes results -> Final Answer
# ============================================================


# STEP 1: IMPORT REQUIRED LIBRARIES

from dotenv import load_dotenv
# Loads environment variables from the .env file.
# We use this to access GOOGLE_API_KEY securely.


from langchain_google_genai import ChatGoogleGenerativeAI
# Connects LangChain with Google's Gemini model.
# Gemini is the language model (LLM) that understands
# questions, selects tools, and generates answers.


from langchain.tools import tool
# The @tool decorator converts a Python function into
# a tool that a LangChain agent can call.


from langchain.agents import create_agent
# Current agent API in LangChain 1.x.
# It creates an agent that can call tools and process
# their results without manually creating AgentExecutor.


from ddgs import DDGS
# Standalone DuckDuckGo search library.
# This avoids importing DuckDuckGoSearchRun from the
# deprecated langchain-community integration.


# STEP 2: LOAD ENVIRONMENT VARIABLES

load_dotenv()

# WHY?
# Loads values from .env into the environment.
# Add GOOGLE_API_KEY=your_actual_key to your .env file.
# Never hardcode or share your actual API key.


# STEP 3: INITIALIZE THE LANGUAGE MODEL

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

# Gemini is the BRAIN of the agent.
# It understands the question and decides whether to search.
# It also uses search results to generate the final answer.
# The model itself does not automatically browse the web.


# STEP 4: CREATE A CUSTOM INTERNET SEARCH TOOL

@tool
def web_search(query: str) -> str:
    """Search the internet for information about a query."""

    # DDGS performs the actual internet search.
    # max_results=5 limits the number of results returned.
    results = DDGS().text(query, max_results=5)

    # Handle the case where no results are returned.
    if not results:
        return "No search results found."

    # Convert search results into readable text.
    # The agent can pass this information back to Gemini.
    return "\n\n".join(
        f"Title: {item.get('title', '')}\n"
        f"URL: {item.get('href', '')}\n"
        f"Snippet: {item.get('body', '')}"
        for item in results
    )


# OPTIONAL: TEST THE SEARCH TOOL BY ITSELF

# results = web_search.invoke(
#     {"query": "top news in India today"}
# )
# print(results)

# Testing the tool separately helps us identify search
# errors before debugging the agent.


# STEP 5: CREATE THE AGENT

agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=(
        "You are an internet research assistant. "
        "Use web_search when external information is needed. "
        "Give clear answers based on the search results. "
        "Do not invent facts when information is uncertain."
    )
)

# model:
# Supplies Gemini as the language model.
#
# tools:
# Gives the agent access to our custom web search function.
#
# system_prompt:
# Provides instructions that guide the agent's behavior.
#
# create_agent:
# Creates the agent workflow. The agent can call the tool,
# receive observations, and continue toward a final answer.


# STEP 6: SEND THE USER'S QUESTION

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "3 ways to reach Goa from Delhi"
            }
        ]
    }
)

# invoke() runs the agent workflow.
#
# The new API uses a messages list instead of the classic
# {"input": "..."} format.
#
# The agent may search the web, process observations,
# and produce a final response.


# STEP 7: PRINT THE FINAL ANSWER

print(response["messages"][-1].content)

# response["messages"] contains the conversation messages.
# The final message is normally the agent's answer.
# .content extracts its text.

# Extract the final message from the agent
final_message = response["messages"][-1]

# Get its content
content = final_message.content

# Handle both plain text and structured text content
if isinstance(content, list):
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            print(item.get("text", ""))
else:
    print(content)