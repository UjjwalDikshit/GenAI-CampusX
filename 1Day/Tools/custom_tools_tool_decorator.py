from langchain_core.tools import tool

# Steps:

# create a function 
# highly recommended to add doc-string
# must write type-hinting so that llm can know which type of data to return
# add @tool

@tool # -> make this function special function, now llm can communicate with this
def multiple(a:int, b:int) -> int:
    """ Multiply two numbers """
    return a*b

# use this tool -> runnable hai

result = multiple.invoke({"a":3,"b":5})
print(result)


print(multiple.name) # special function has these attributes also
print(multiple.description)
print(multiple.args)

# if we pass this tool to llm -> what it see

print(multiple.args_schema.model_json_schema()) # llm see json string, logic into json

