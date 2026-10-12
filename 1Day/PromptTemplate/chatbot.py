import os
import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

os.environ["HF_HOME"] = "D:/Huggingface_cache"

print("Loading model into VRAM/RAM... (This happens only ONCE)")

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen3-0.6B",
    task="text-generation",
    device_map="auto",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.5,
    ),
    model_kwargs=dict(
        dtype=torch.float16,
    ),
)

model  = ChatHuggingFace(llm = llm)

# while True:
#     user_input = input("You: ")
#     if user_input == 'exit':
#         break
#     result = model.invoke(user_input)
#     print("AI: ", result)


#-----------------------------------------------------------
# require to_make chat_hisory list so that llm also has context of previous conversation.
# put prompt and response in this list

chat_history = [
    SystemMessage(content = "You are a powerful assistment tool")
]

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))

    response = model.invoke(chat_history)

    print("Raw Response:", response)
    print("Content:", response.content)

    chat_history.append(response)
    
print(chat_history)

# here chat_history has all the messages, but who has sended what, role for each message is missing - problematic.

