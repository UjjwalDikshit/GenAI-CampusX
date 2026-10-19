from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

loader = TextLoader('cricket.txt',encoding='utf-8')

docs = loader.load()

#--------------------------------------------------------------------------
# print(docs)
# print(type(docs)) # python list, list of document object.
# print(docs[0])
# #print(docs[1])  not working as text is small so it comes in single list

# print(docs[0].page_content)
# print(docs[0].metadata)

#--------------------------------------------------------------------------

parser = StrOutputParser()

template = PromptTemplate(
    template = "Write some facts on the basis of these text \n {text}",
    input_variables=['text']
)

chain = template | model | parser

result = chain.invoke({'text':docs[0].page_content})

print(result)
