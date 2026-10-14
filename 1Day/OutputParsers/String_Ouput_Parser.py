from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate # to make dynamic prompt here imported this.
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')


# 1st promt -> detail report

promtOne = PromptTemplate(
    template = "Write a detailed report on {Topic}",
    input_variables=['Topic']
)


# 2nd prompt -> summary


promtTwo = PromptTemplate(
    template = "Write a 5 line summary on following text \n {Text}",
    input_variables=['Text']
)

# ---------------------------------------------------------------


# prmt1 = promtOne.invoke({"Topic": "Black Hole"})

# prmt2 = promtTwo.invoke({"Text": respone1.content})

# respone1 = model.invoke(prmt1)

# respone2 = model.invoke(prmt2)

# print(respone2.content)


# ---------------------------------------------------------------------
# with help of sTring output parser

parser = StrOutputParser()

chain = promtOne | model | parser | promtTwo | model | parser

result = chain.invoke({'Topic':"Blackhole"})

print(result)
