from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
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

chain=RunnableSequence(prompt1,model,parser,prompt2,model,parser)
                                        
# LECL=prompt1 | model | parser | prompt2 | model | parser

result=chain.invoke({"topic":"Fun"})
# Here only the explanation will be printed if yo also want to print joke than use RunnablePassThrough
print(result)

