from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

prompt1=PromptTemplate(
        template="Write a Joke about the {topic}",
        input_variables=['topic'],
 )
prompt2=PromptTemplate(
        template="Explain the following Joke {text}",
        input_variables=['text'],
 )
model = ChatHuggingFace(llm=llm)

parser=StrOutputParser()

joke_chain=RunnableSequence(prompt1,model,parser)

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2,model,parser)
})

final_chain=RunnableSequence(joke_chain,parallel_chain)
result=final_chain.invoke({'topic':'running'})

print(result)
print(result['joke'])
print(result['explanation'])