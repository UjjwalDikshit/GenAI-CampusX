from langchain_openai import ChatOpenAI
from dotenv import load_dontenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dontenv()

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI() # default model will get selected, as not passed anything
parser = StrOutputParser()
chain = prompt | model | parser
result = chain.invoke({'topic':'cricket'}) # pass input of first step
print(result)

# To visualise chain
chain.get_graph().print_ascii()