from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int

new_person:Person = {'name':'ujjwal','age':34}
print(new_person)

# created data format.

# i've reviews of mobile
# i wanted to send those to llm and take out {summary,sentiment,rating} for each review
#

