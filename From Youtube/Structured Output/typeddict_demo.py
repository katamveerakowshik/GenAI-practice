from typing import TypedDict

class Person(TypedDict):
    name : str
    age : int

# new_person = Person(name="John", age=30)
# {'name': 'John', 'age': 30}

new_person = Person(name="John", age="30")
# {'name': 'John', 'age': '30'}

print(new_person)