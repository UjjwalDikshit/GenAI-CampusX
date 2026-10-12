from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()
model = ChatOpenAI()

message=[
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content = "Tell me about Langhchain")
]

result = model.invoke(message)
message.append(result) # result is in alredy AIMessage format
print(message)