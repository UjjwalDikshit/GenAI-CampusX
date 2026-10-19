from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

loader = PyPDFLoader('civic_pulse_plan.pdf')

docs = loader.load()

#----------------------------------------------------------------------------------

# print(len(docs))
# print(docs)

# print(docs[0].page_content)
# print(docs[1].metadata)
#----------------------------------------------------------------------------------

# parser = StrOutputParser()

# template = PromptTemplate(
#     template = "Write some facts on the basis of these text \n {text}",
#     input_variables=['text']
# )

# chain = template | model | parser

# result = chain.invoke({'text':docs[0].page_content})

# print(result)
