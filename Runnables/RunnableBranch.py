from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableLambda,RunnableBranch
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

prompt1=PromptTemplate(
        template="Write a detailed report about the {topic}",
        input_variables=['topic'],
 )
prompt2=PromptTemplate(
        template="Summarize the following {text}",
        input_variables=['text'],
 )

model = ChatHuggingFace(llm=llm)

parser=StrOutputParser()

joke_chain=RunnableSequence(prompt1,model,parser)

branch_chain=RunnableBranch(
    (lambda x : len(x.split())>500,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain=RunnableSequence(joke_chain,branch_chain)
result=final_chain.invoke({'topic':'How AI has Grown over the years'})

print(result)
