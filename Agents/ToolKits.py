## Collection of tools
from langchain_community.tools import tool

@tool
def addition(a:int,b:int,c:int) -> int :
    """Returns add of the number"""
    return a+b

@tool 
def loops(list):
    """return the list of some values"""
    a=[]
    for l in list:
        a.append(l)

    return l

@tool
def decide(dnry):
    """decides between the dictionary"""
    dt={}

    for d in dnry:
        dt[d]=d.value

    return dt

class ToolKit:
    def get_tools(self):
        return [addition,loops,decide]
    
toolkit=ToolKit()
for t in toolkit:
    print(t.name,"|",t.description)
