# lanchain is umbrella library, langchain_core keeps frequenlty used component
# this code run for langhcain 0.x, not for langchain 1.x

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

# First made schema => guide llm to make ouput or that type , make using responseSchema
# in responseSchema we send list of schmea object.

schema = [
    ResponseSchema(name="fact_1", description="fact1 tell about the three deep concept of normal  psychology"),
    ResponseSchema(name="fact_2", description="fact2 tell about the three deep concept of poor  psychology"),
    ResponseSchema(name="fact_3", description="fact3 tell about the three deep concept of rich  psychology"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions}
)


# prompt = template.invoke({'topic':'girl'})
# result = model.invoke(prompt)
# final_result= parser.parse(result.content)
# print(final_result)


chain = template | model | parser
result = chain.invoke({'topic':'boy'})
print(result)

# here we can't do data validation => that's why we pydantic sturctured parser.