from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

prompt=PromptTemplate(
        template="Write a Joke about the {topic}",
        input_variables=['topic'],
 )

# def words(text):
#     return len(text.split())

model = ChatHuggingFace(llm=llm)

parser=StrOutputParser()

joke_chain=RunnableSequence(prompt,model,parser)

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    # 'words':RunnableLambda(words)
    'words':RunnableLambda(lambda x: len(x.split()))
})

final_chain=RunnableSequence(joke_chain,parallel_chain)
result=final_chain.invoke({'topic':'acting'})

final_result="""{} \n word count : {}""".format(result['joke'],result['words'])
print(final_result)
