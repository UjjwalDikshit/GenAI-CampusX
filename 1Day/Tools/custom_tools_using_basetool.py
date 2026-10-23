# basetool class -> has knowledge ki ek tool has how to behave in langchain
# create using directly inheriting basetool


from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel,Field

# arg schema using pydantic

class MultiplyInput(BaseModel): # defined here argument schema, using pydantic
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")


class MultiplyTool(BaseTool): #making tool class by directly iheriting basetool
    name: str = "multiply"
    description: str = "Multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int: # _run -> can't change this.
        return a * b

multiply_tool = MultiplyTool()

result = multiply_tool.invoke({'a':3, 'b':3})

print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)
