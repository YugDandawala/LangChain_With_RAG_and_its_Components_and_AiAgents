from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

prompt1=PromptTemplate(
        template="Generative a detailed report on topic {topic}",
        input_variables=['topic']
    )
prompt2=PromptTemplate(
        template="Generative a 5 line summary from the following text \n{text}",
        input_variables=['text']
    )
model = ChatHuggingFace(llm=llm)
parser=StrOutputParser()

chain=prompt1 | model | parser | prompt2 | model | parser # LCEL
result=chain.invoke({'topic':'The rise of Artificial Intelligence'})
print(result)

# To visually see the graph of chain
chain.get_graph().print_ascii()