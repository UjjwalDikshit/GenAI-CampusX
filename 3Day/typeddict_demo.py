from typing import TypedDict

class Person(TypedDict):
    name:str
    age: int

new_person: Person = {'name':"Ujjwal",'age':32,'gender':'male'}
print(new_person)