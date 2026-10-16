from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

parser = StrOutputParser()

template1 = PromptTemplate(
    template="Generate 5 interesting psychological facts about {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Generate 5 summary about following text \n {text}",
    input_variables=['text']
)

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'human'})

print(result)

chain.get_graph().print_ascii()


















