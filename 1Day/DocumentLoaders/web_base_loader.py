# if you want to query over content from website
# work with most static website, for javascript heavy website use selenium website 
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

url= "https://reference.langchain.com/python/langchain-openai"

loader = WebBaseLoader(url) # u'can send list of url
docs = loader.load()

#------------------------------------------------

# print(len(docs)) # 1doc/url

# print(docs[0].page_content)

#------------------------------------------------

parser = StrOutputParser()

template1 = PromptTemplate(
    template="Frame me 2 interview hardest problem from this text \n {text}",
    input_variables=['text']
)

template2 = PromptTemplate(
    template="Answer the following question : {question} from this text \n {text}",
    input_variables=['text','question']
)

chain = template2 | model | parser
print(chain.invoke({'question':'summarise this page','text':docs[0].page_content}))
