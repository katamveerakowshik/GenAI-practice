from pydantic import BaseModel, EmailStr, Field
from typing import Optional

"""When using Optional inside pydantic, it should always be given some value"""
class Student(BaseModel):
    name: str
    age: int
    address: Optional[str] = "None"
    email: EmailStr
    cgpa: float = Field(ge=0.0, le=10.0, description="cgpa should be between 0.0 and 10.0", default=5.0)


# new_student = Student(name=32, age=30, address="123 Main St", email="9yOJ1@example.com", cgpa=8.5)
#Throws error as name is not a string

# new_student = Student(name="John", age=30, email="9yOJ1@example.com", cgpa=8.5)
#Runs and gives address as none

# new_student = Student(name="John", age=30, address = "123 Main St", email="9yOJ1@", cgpa=8.5)
#Throws an error saying invaid email address

# new_student = Student(name="John", age=30, address="123 Main St", email="9yOJ1@example.com", cgpa=12)
#Throws an error as cgpa is not between 0.0 and 10.0

new_student = Student(name="John", age=30, address="123 Main St", email="9yOJ1@example.com", cgpa=8.5)

print(new_student)
