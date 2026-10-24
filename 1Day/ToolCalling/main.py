from dotenv import load_dotenv
from langchain_core.tools import StructuredTool,tool
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field
import requests
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
import sys


load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

# Define the tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers and return the result."""
    return a * b



# result = multiply.invoke({"a": 10, "b": 20})
# print(result)

# print(multiply.name)
# print(multiply.description)
# print(multiply.args)


# each llm has not ability to bind with tools

# Bind the tool to the model
llm_with_tools = model.bind_tools([multiply])

messages = [HumanMessage("Multiply 3 and 5")]

# Invoke the model
result = llm_with_tools.invoke(messages) # AI result
messages.append(result)

# Inspect the response
print("Content:", result.content)
print("Tool calls:", result.tool_calls)

# Execute the tool if the model requests it
# for tool_call in result.tool_calls:
#     if tool_call["name"] == "multiply":
#         output = multiply.invoke(tool_call["args"])
#         print("Tool output:", output)

# {'name': 'multiply', 'args': {'a': 3, 'b': 5}, 'id': '067efb48-c27d-48a9-be4f-9cafd360cbd3', 'type': 'tool_call'}
# tool call by llm

newresult = result.tool_calls[0]
print(multiply.invoke(newresult['args'])) # but only return result
print(multiply.invoke(newresult)) # result + extra content in proper package -> in ToolMessage (special messsge, while exectuing msg)
print(type(multiply.invoke(newresult)))
# now send it to llm 

toolresult = multiply.invoke(newresult)

messages.append(toolresult)  # making conversation history

final_result = llm_with_tools.invoke(messages)
print(messages)
print(final_result.content)

