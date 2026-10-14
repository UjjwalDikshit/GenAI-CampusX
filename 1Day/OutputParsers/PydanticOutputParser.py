# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser
# from pydantic import BaseModel,Field

# from dotenv import load_dotenv
# load_dotenv()

# model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')



# # create pydantic object work as schema

# class Person(BaseModel):
#     name: str = Field(description="Name of the person")
#     age: int = Field(gt = 18, description="Age of the person")
#     city : str = Field(description="Name of the city the person belogns ")

# parser = PydanticOutputParser(pydantic_object= Person)

# template = PromptTemplate(
#     template='Generate the name age and city of a fictional {place} person \n {format_instruction}',
#     input_variables=['place'],
#     partial_variables={'format_instruction':parser.get_format_instructions()}
# )

# prompt = template.invoke({"place":"italy"})
# print(prompt)
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Load environment variables if you have a .env file containing GOOGLE_API_KEY
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# Create Pydantic schema
class Person(BaseModel):
  name: str = Field(description="Name of the person")
  age: int = Field(gt=18, description="Age of the person")
  city: str = Field(description="Name of the city the person belongs to")


# Setup parser and prompt template
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template=(
        "Generate the name, age, and city of a fictional {place} person.\n"
        " {format_instruction}"
    ),
    input_variables=["place"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    },
)

# Combine into a single chain
chain = template | model | parser

# Invoke the chain directly with the input dictionary
final_result = chain.invoke({"place": "italy"})

# final_result is already parsed into your Pydantic object
print(final_result)
print(type(final_result))
print(f"Name: {final_result.name}, Age: {final_result.age}, City: {final_result.city}")