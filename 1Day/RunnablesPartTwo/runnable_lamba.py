# this runnable help convert any python function into runnable.

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough,RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')


def word_counter(text):
    return len(text.split())

runnable_word_counter = RunnableLambda(word_counter)
#---------------------------------------------------------------------------
# result = runnable_word_counter.invoke('hi there how are you?')
# print(result)
#---------------------------------------------------------------------------

prompt1 = PromptTemplate(
    template= 'Write a joke about given {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

joke_generator_chain = RunnableSequence(prompt1, model, parser)

chain = RunnableParallel(
    {
    'joke':RunnablePassthrough(),
    'count_of_words':RunnableLambda(word_counter)
    }
)

final_chain = RunnableSequence(joke_generator_chain,chain)
final_result = final_chain.invoke({'topic':'Cricket'})

print("joke: \n" + final_result['joke'])
print(" ")
print("count_of_words: \n", final_result['count_of_words'])