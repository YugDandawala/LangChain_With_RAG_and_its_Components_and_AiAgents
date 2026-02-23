from langchain_community.tools import tool
from langchain.messages import HumanMessage
from langchain_ollama import OllamaLLM
from langchain_classic.agents import initialize_agent,AgentType

@tool
def multiply(a:int,b:int) -> int:
    """Multiply 2 numbers"""
    return a*b

@tool
def add(a:int,b:int) -> int:
    """Addition of 2 numbers"""
    return a+b

model = OllamaLLM(
    model = "llama3.2",
    temperature = 0.1
)

#Tool Binding used for simple and single model setup not recommended(works with list llm)
# --> llm_with_tools=model.bind_tools([multiply])

# Initialize robust agent
agent = initialize_agent(
    tools = [multiply, add],   # multiple tools
    llm = model,            # can integrate multi-model orchestration

    agent = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, #an object that decides how to use tools + LLM reasoning.
    # (zero-shot-react-description/zero-shot-react-docstore/conversational-react-description)
    # (self-ask-with-search/structured-chat-zero-shot/reAct-style (classic ReAct)
    # (conversational-react-docstore/self-ask-with-search-plus-memory/structured-chat-conversational)
    
    verbose = True #debbugging
)

query = HumanMessage("Multiply 7 and 8 and add result with 4")
messages = [query]
print(agent.invoke(messages))