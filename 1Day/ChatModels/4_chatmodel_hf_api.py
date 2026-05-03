from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint( # which model you wnat to use
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token="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"

)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)


