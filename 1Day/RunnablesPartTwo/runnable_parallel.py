from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableSequence

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

prompt1 = PromptTemplate(
    template= 'Write a tweet about {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template= "Write linkedin post about {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = RunnableParallel({
    'tweet':RunnableSequence(prompt1, model, parser),
    'linkedin':RunnableSequence(prompt2, model, parser)
    }
)

result = chain.invoke({'text':'AI'})

print(result['tweet'])
print(" ")
print(result['linkedin'])


