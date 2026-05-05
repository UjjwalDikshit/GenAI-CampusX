from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str
    roll :str = "2023BITE044" #default values
    caste:Optional[str] = "Ku-jaat"
    email: EmailStr
    cgpa : float= Field(gt=0,lt=10, default=4,description='A description value repsrenting the cgpa of the student') # putting constraints


new_student = {'name':"Ujjwal Dikshit"}
new_student = {'name':"Ujjwal Dikshit",'caste':'44','email':'ujjwaldikshit1@gmail.com'}

student = Student(**new_student)   # this is pydanctic object, also can be converted into python dict-object

student_dict = dict(student)
print(student_dict['age'])

print(student)
print(student.name) 

student_json = student.model_dump_json()