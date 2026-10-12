from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()

llm = HuggingFaceEndpoint( # which model you want to use
    repo_id="Qwen/Qwen3-0.6B",
    task="text-generation",
    # huggingfacehub_api_token="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"
    # huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)


