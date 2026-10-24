from dotenv import load_dotenv
from langchain_core.tools import StructuredTool,tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage
import requests
import os

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

# @tool
# def multiply(a: float, b: float) -> float:
#     """Multiply two numbers and return the result."""
#     return a * b

# @tool
# def get_conversation_factor(base_currency: str, target_currency:str)->float:
#     """This function fetches the curreny conversation factor between a given base currency and a target currency"""
#     url = f'https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}'
#     response = requests.get(url, timeout=10)
#     response.raise_for_status()
#     data = response.json()
#     if data.get("result") != "success":
#         raise ValueError(f"API error: {data}")

#     return float(data["conversion_rate"])


#-----------------------------------------------------------------------


load_dotenv()



@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b


@tool
def get_conversion_factor(
    base_currency: str,
    target_currency: str
) -> float:
    """Fetch the exchange rate between two currencies."""
    api_key = os.getenv("EXCHANGE_RATE_API_KEY")

    if not api_key:
        raise ValueError("EXCHANGE_RATE_API_KEY is missing from .env")

    url = (
        f"https://v6.exchangerate-api.com/v6/{api_key}"
        f"/pair/{base_currency.upper()}/{target_currency.upper()}"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get("result") != "success":
        raise ValueError(f"Exchange rate API error: {data}")

    return float(data["conversion_rate"])


tools = [multiply, get_conversion_factor]
tool_map = {t.name: t for t in tools}

llm_with_tools = model.bind_tools(tools)

# messages = [
#     HumanMessage(
#         content="Convert today's 1000 USD into Indian rupees. "
#                 "Fetch the current exchange rate and calculate the result."
#     )
# ]

messages = [
    HumanMessage(
        content="What is the conversion factor between USD and INR, and based on that can you convert 10 usd to inr."
    )
]

# Run the tool-calling loop
for _ in range(5):
    result = llm_with_tools.invoke(messages)
    messages.append(result)

    if not result.tool_calls:
        print("Final response:", result.content)
        break

    for tool_call in result.tool_calls:
        selected_tool = tool_map[tool_call["name"]]

        try:
            output = selected_tool.invoke(tool_call["args"])
        except Exception as exc:
            output = f"Tool execution failed: {exc}"

        print(f"{tool_call['name']} -> {output}")

        messages.append(
            ToolMessage(
                content=str(output),
                tool_call_id=tool_call["id"]
            )
        )
else:
    print("Stopped: maximum tool-calling rounds reached.")
