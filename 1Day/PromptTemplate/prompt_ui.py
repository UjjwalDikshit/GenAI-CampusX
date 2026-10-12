import os
os.environ["HF_HOME"] = "D:/Huggingface_cache"


from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import torch



load_dotenv()

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

# -------------------------------------------------------------
# st.header('Research Tool')
# user_input = st.text_input("Enter your prompt")

# if st.button("Summarize"):
#     result = llm.invoke(user_input)
#     # st.text("Some random text")
#     print(result)
#     st.write(result)

# -------------------------------------------------------------

st.header('Research Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Select ...", "Attention Is All You Need",
"BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5paragraphs)", "Long (detailed explanation)"] )

#making prompt template
template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following
specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}
1. Mathematical Details.
- Include relevant mathematical equations if present in the paper.
- Explain the mathematical concepts using simple, intuitive code snippets
where applicable.
2. Analogies:
- Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with: "Insufficient
information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and
length.
"
""",
input_variables=['paper_input','style_input','length_input']
)

# use saved template here without writing upper one, if u've any
# template = load_prompt("template.json")

print("")
print("")
print(template)

#fill the placeholders
prompt = template.invoke(
    {'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
    }
)



print("")
print("")
print(prompt)

if st.button('Summarize'):
    st.write("Hello,")
    result = llm.invoke(prompt)
    st.write(result)


# -------------------------------------------------------------
