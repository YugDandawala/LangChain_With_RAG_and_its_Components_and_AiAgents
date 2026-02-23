from langchain_community.tools import StructuredTool,BaseTool,tool
from pydantic import Field,BaseModel
from typing import Type

# 1st Method to define tool

@tool
def multiply(a:int,b:int) -> int:
    """Multiply 2 numbers"""
    return a*b

class Multiplyinput(BaseModel):
    a : int = Field(required=True,description="First number to add")
    b : int = Field(required=True,description="Second number to add")

def multiply(a:int,b:int) -> int:
    return a*b

# 2nd Method to define tool

multiply_tool=StructuredTool.from_function(
    func=multiply,
    name="multiply agent",
    description="Multiplication",
    args_schema=Multiplyinput
)

#3rd method to define tool
class multiplytool(BaseTool):
    name:str="Multiply"
    description:str="Multiple 2 number"

    args_schema:Type[BaseModel]=Multiplyinput

    def _run(self, a:int, b:int): #name needs to be _run only
        return a * b

multiply=multiplytool()
result=multiply.invoke({"a":5,"b":6})

print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)