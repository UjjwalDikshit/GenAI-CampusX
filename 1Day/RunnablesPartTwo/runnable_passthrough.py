from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

prompt1 = PromptTemplate(
    template= 'Write a joke about given {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template= "Write explanation of given jokes \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

joke_generator_chain = RunnableSequence(prompt1, model, parser)

chain = RunnableParallel(
    {
    'joke':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2, model, parser)
    }
)

final_chain = RunnableSequence(joke_generator_chain,chain)
final_result = final_chain.invoke({'topic':'Cricket'})

print(final_result['joke'])
print(" ")
print(final_result['explanation'])


#-------------------------------------------------------------------


# passthrough = RunnablePassthrough()

# print(passthrough.invoke({'name':'Ujjwal Dikshit'})) # will return same dictionary as output


