#2

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

# loader = PyPDFLoader(file_path='jee main score card.pdf')
loader = PyPDFLoader('jee main score card.pdf') # both works

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=40,
    chunk_overlap=0,
    separator='',
)

parser = StrOutputParser()
#----------------------------------------------------------------
# print(splitter.split_text(docs[0].page_content))
# print("")
# print("Gemini call:====)")
#----------------------------------------------------------------
result = splitter.split_documents(docs) # list of chunk, each chunk is document object has metadata, page_content 
print(result[1]) # each chunk here is doc object
print("")
print("Gemini call:====)")
#----------------------------------------------------------------
template = PromptTemplate(
    template="summarise this result {text}",
    input_variables=['text']
)

chain = template | model | parser

# result = chain.invoke({'text':docs[0].page_content}) # we are here not sending splitted docs, but content from document loader page-1

result = chain.invoke({'text':result[1].page_content})

print(result)