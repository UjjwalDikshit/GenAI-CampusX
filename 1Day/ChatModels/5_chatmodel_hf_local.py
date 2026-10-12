# import os

# os.environ["HF_HOME"] = "D:/Huggingface_cache"

# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# llm = HuggingFacePipeline.from_model_id(

#     model_id="Qwen/Qwen3-0.6B",
#     task="text-generation",
#     pipeline_kwargs=dict(
#         max_new_tokens=512,
#         do_sample=False,
#         repetition_penalty=1.03,
#     ),
# )

# model = ChatHuggingFace(llm=llm)

# result = model.invoke("5 dj star")
# print(result.content)

#------------------------------------------------------------------------------------------------------------------------------

# import os

# os.environ["HF_HOME"] = "D:/Huggingface_cache"

# from huggingface_hub import snapshot_download
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
# CACHE_DIR = "D:/Huggingface_cache/hub"

# # Reuse cached files and download anything missing
# model_path = snapshot_download(
#     repo_id=MODEL_ID,
#     cache_dir=CACHE_DIR,
# )

# print("Model files available at:", model_path)

# # Load the downloaded model locally
# llm = HuggingFacePipeline.from_model_id(
#     model_id=model_path,
#     task="text-generation",
#     pipeline_kwargs={
#         "max_new_tokens": 512,
#         "do_sample": False,
#         "repetition_penalty": 1.03,
#     },
# )

# model = ChatHuggingFace(llm=llm)

# result = model.invoke("What is the capital of India?")
# print(result.content)
# --------------------------------------------------------------------------------------------------------------
# import os
# import torch

# os.environ["HF_HOME"] = "D:/Huggingface_cache"

# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# llm = HuggingFacePipeline.from_model_id(
#     model_id="Qwen/Qwen3-0.6B",
#     task="text-generation",
#     device_map="auto",  # Loads the model directly onto GPU
#     pipeline_kwargs=dict(
#         max_new_tokens=512,
#         do_sample=True,         # Sampling is REQUIRED for Qwen3 to work well
#         temperature=0.7,        # Qwen's recommended non-thinking temperature
#         top_p=0.8,              # Top-p filtering
#         repetition_penalty=1.5, # Prevents small model looping
#         # torch_dtype=torch.float16,  Halves VRAM usage (~1.2 GB required)
#     ),
#     # Use `dtype` instead of the deprecated `torch_dtype`
#     model_kwargs=dict(
#         dtype=torch.float16,
#     ),
# )

# model = ChatHuggingFace(llm=llm)

# result = model.invoke("Tell me about tradition of delhi different from india")
# print(result.content)

# -------------------------------------------------------------------------------------------------------

import os
import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

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

model = ChatHuggingFace(llm=llm)

print("\nModel ready! Type 'exit' to quit.\n" + "-" * 40)

# Continuous loop keeps the model loaded in RAM/VRAM
while True:
    user_input = input("\nYou: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        break
    
    result = model.invoke(user_input)
    print(f"\nQwen: {result.content}")