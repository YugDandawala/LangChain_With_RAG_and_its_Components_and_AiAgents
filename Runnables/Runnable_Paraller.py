from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

prompt1=PromptTemplate(
        template="Generate a tweet about {topic}",
        input_variables=['topic'],
 )
prompt2=PromptTemplate(
        template="Generate Linkedin post about {topic}",
        input_variables=['topic'],
 )
model = ChatHuggingFace(llm=llm)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'tweet':RunnableSequence(prompt1,model,parser),   #chain1 = prompt1 | model |parser its same only 
    'linkedin':RunnableSequence(prompt2,model,parser),
})

result=parallel_chain.invoke({'topic':'sport'})
print(result)
print(result['tweet'])
print(result['linkedin'])