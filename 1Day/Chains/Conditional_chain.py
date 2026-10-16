from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableBranch # this help make branching
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from pydantic import Field,BaseModel
from typing import Literal


load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

class Feedback(BaseModel):
    sentiment:Literal['positive','negative'] = Field(description="Give the sentiment of the feedback")

parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment into positive or negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

# classifier_chain = promtp1 | model | parser # no gurantee that llm return only that two words, can return sentence too. On this word our next part depend so must be from that words. so use pydantic output parser.
classifier_chain = prompt1 | model | parser2 # this will always return 'positive' or 'negative'

# result = classifier_chain.invoke({'feedback':"I have not seen this worst phone in ny life, apple ki ma ki chut"})

# making second part of chain
# RunnableBranch -> pass tuple => condition and which chain to run on getting condition true.

# branch_chain = RunnableBranch(
#     (condition1,chain1 ),
#     (condition2,chain2 ),
#     RunnableLambda(default chain) # simple run lambda function -> but this is not chain so convert it into runanble 
# )

prompt2 = PromptTemplate(
    template="""
Write a professional and polite response to the following positive customer feedback:

Feedback: {feedback}
""",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="""
Write a professional and empathetic response to the following negative customer feedback. Acknowledge the customer's concern and apologize if appropriate.

Feedback: {feedback}
""",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive',prompt2 | model | parser1 ),
    (lambda x: x.sentiment == 'negative',prompt3 | model | parser1 ),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain
result = chain.invoke({'feedback':"I have not seen this worst phone in ny life, apple ki ma ki chut"})

print(result)
