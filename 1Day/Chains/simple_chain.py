from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

parser = StrOutputParser()

template = PromptTemplate(
    template="Generate 5 interesting psychological facts about each of rich, poor, midlle class {gender}",
    input_variables=['gender']
)

chain = template | model | parser

result = chain.invoke({'gender':'female'})

print(result)

# can print steps taken in chain exection like this

chain.get_graph().print_ascii()
