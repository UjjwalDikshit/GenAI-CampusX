from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough,RunnableSequence,RunnableBranch

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

prompt1 = PromptTemplate(
    template= 'Write a detailed report on {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template= "summarize the following text \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(     # here we pass tuples, holding condition : runnable, and at last there is a default
    (lambda x: len(x.split())>500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough() # default
)

final_chain = RunnableSequence(report_gen_chain,branch_chain)

print(final_chain.invoke({'text':'Russia vs Ukraine'}))


