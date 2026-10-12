# import torch
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     BitsAndBytesConfig,
# )

# MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )

# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     quantization_config=quantization_config,
#     device_map="auto",
#     low_cpu_mem_usage=True,
# )

# messages = [
#     {"role": "user", "content": "Explain LangChain in three sentences."}
# ]

# prompt = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
# )

# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# with torch.inference_mode():
#     output = model.generate(
#         **inputs,
#         max_new_tokens=100,
#         do_sample=True,
#         temperature=0.7,
#     )

# print(tokenizer.decode(output[0], skip_special_tokens=True))



# import os

# os.environ["HF_HOME"] = "D:/Huggingface_cache"
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# import torch
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     BitsAndBytesConfig,
# )

# MODEL_PATH = (
#     "D:/Huggingface_cache/hub/models--HuggingFaceH4--zephyr-7b-beta/"
#     "snapshots/892b3d7a7b1cf10c7a701c60881cd93df615734c"
# )

# print("CUDA available:", torch.cuda.is_available())
# print("GPU:", torch.cuda.get_device_name(0))

# # Load the tokenizer from the existing local files
# tokenizer = AutoTokenizer.from_pretrained(
#     MODEL_PATH,
#     local_files_only=True,
# )

# # Quantize model weights to 4-bit
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )

# print("Loading Zephyr in 4-bit mode...")

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     quantization_config=quant_config,
#     device_map="auto",
#     low_cpu_mem_usage=True,
#     local_files_only=True,
# )

# print("Model loaded successfully!")

# prompt = """<|user|>
# What is the capital of India?</s>
# <|assistant|>
# """

# inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

# with torch.inference_mode():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=80,
#         do_sample=False,
#     )

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

MODEL_PATH = (
    r"D:\Huggingface_cache\hub\models--HuggingFaceH4--zephyr-7b-beta"
    r"\snapshots\892b3d7a7b1cf10c7a701c60881cd93df615734c"
)

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

print("Loading model with CPU offloading...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quant_config,
    device_map="auto",
    max_memory={0: "3GiB", "cpu": "12GiB"},
    low_cpu_mem_usage=True,
    local_files_only=True,
)

messages = [
    {"role": "user", "content": "Explain LangChain in three sentences."}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt")
input_device = model.get_input_embeddings().weight.device
inputs = {key: value.to(input_device) for key, value in inputs.items()}

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))