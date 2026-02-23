from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.runnables import RunnableBranch,RunnableLambda
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm)

parser=StrOutputParser()

class Feedback(BaseModel):
        sentiment:Literal['Positive',"Negative"]=Field(description='GIve the sentiment of the Feedback')
        
parser2=PydanticOutputParser(pydantic_object=Feedback)
        
prompt1=PromptTemplate(
        template="Classify the sentiment of the following feedbaack text into postive or negative \n{feedback} \n {format_instruction}",
        input_variables=['feedback'],
        partial_variables={'format_instruction':parser2.get_format_instructions()}
 )
        
prompt2=PromptTemplate(
        template="Write an positive response to the feedback \n {feedback}",
        input_variables=['feedback'],
 )

prompt3=PromptTemplate(
        template="Write an negative response to the feedback \n {feedback}",
        input_variables=['feedback'],
 )

classifier_chain = prompt1 | model1 | parser2

branch_chain=RunnableBranch(
    (lambda x : x.sentiment=='Positive',prompt2 | model1 | parser),
    (lambda x : x.sentiment=='Negative',prompt3 | model1 | parser),
    # this should be a default chain
   RunnableLambda( lambda x:"Could not find the sentiment")
)

chain= classifier_chain | branch_chain

result=chain.invoke({'feedback':"This is a very wonderful phone"})
print(result)

# To visually see the graph of chain
chain.get_graph().print_ascii()