from langchain_openai import ChatOpenAI  #this is chatmodel
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model = 'gpt-4',  temperature = 0.3, max_completion_tokens = 10)

result = model.invoke('What is capital of India')
print(result)
print(result.content)

# here result is not simple plain text