from pydantic import BaseModel,Field, EmailStr # builtin validation
from typing import Optional

# with help of pyndatic you can verify your email
class Student(BaseModel):
    #name:str
    name:str = "Ujjwal" # default value
    age:Optional[int]= None # here default none, if number doesn't come then it will take the place
    email : EmailStr #builtin type
    # cgpa: float = Field(gt= 0, lt = 10) # put constraint,optional to put default
    cgpa: float = Field(gt= 0, lt = 10, default=8, description="A decimal value representing the cgpa of the student") # also can attach regular expression like of phone number

# new_student = {'name':'Ujjwal Dikshit'}
# new_student = {'name':32} will throw error
# new_student = {}
# new_student = {'name':'Ujjwal Dikshit', 'age':23}
# new_student = {'name':'Ujjwal Dikshit', 'age':'23'} # type coerce : sometimes pyndatic see and change implictely input to required input
# new_student = {'name':'Ujjwal Dikshit', 'age':'23','email':"ujjwaldiksthi1@gmail.com",'cgpa':"9"} 
new_student = {'email':"ujjwaldiksthi1@gmail.com"} 


student = Student(**new_student)
print(type(student))
print(student)

#converting to dict
student_dict = dict(student)
print(student_dict['age']) # by doing this i can use this syntax also.

#converting to json
student_json  = student.model_dump_json()
print(student_json)


# Basic Example
# Default values -> if keep standard, if not passed then this will go as arguement
# Optional fields -> mark optional, if not given then fix anything at this place, None or anything
# Coerce
# Builtin validation
# Field Function -> default values, constraints, description, regex (regular expression)
# expressions
# Returns pydantic object -> convert to ison/dict