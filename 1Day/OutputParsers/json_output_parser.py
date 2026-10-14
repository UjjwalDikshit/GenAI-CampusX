from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate # to make dynamic prompt here imported this.
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
parser = JsonOutputParser()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

# here we pass prompt (our demand ) and along with in which format we want out
# output, so prompt text for formatting is not written by us, but there is alreayd
# written prompt available in format_instructions , put in partial variables
# automatically that prompt will get append with passed prompt. and final prompt will go to model

template = PromptTemplate(
    template = "Give me a small imaginary  story , author, country , character name, and character age \n {format_instruction} ",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# this get filled before runtime.
#---------------------------------------------------------------------------------

# prompt = template.format()

# response = model.invoke(prompt)

# # now parse the reponse

# final_result = parser.parse(response.content)

# print(final_result)
# print(type(final_result))

#---------------------------------------------------------------------------------
# writing with help of chain

chain = template | model | parser

response = chain.invoke({}) #if no input then send black dictionary

print(response)
print(type(response))